import {
    ChatCompletionFunctions,
    ChatCompletionRequestMessage,
    ChatCompletionResponseMessage, ChatCompletionResponseMessageRoleEnum,
    Configuration, CreateChatCompletionRequest, CreateImageRequest,
    OpenAIApi
} from "openai";
import {openAILog as log} from "./logging"

import {PluginBase} from "./plugins/PluginBase";
import {AiResponse, MessageData} from "./types";
import {SocksProxyAgent} from "socks-proxy-agent";
import {AxiosRequestConfig} from "axios";

// SOCK proxy configuration
let proxyConfig: AxiosRequestConfig | undefined = undefined

if (process.env["SOCKS_PROXY"]) {
    const proxyAgent = new SocksProxyAgent(process.env["SOCKS_PROXY"])
    proxyConfig = {
        proxy: false,
        httpAgent: proxyAgent,
        httpsAgent: proxyAgent
    }
}

const apiKey = process.env['OPENAI_API_KEY'];
log.trace({apiKey})

const deployment = process.env['OPENAI_DEPLOYMENT_NAME'] ?? 'deployment'
const model = process.env['OPENAI_MODEL_NAME'] ?? 'gpt-3.5-turbo'

// Supported API versions
// https://learn.microsoft.com/en-US/azure/ai-services/openai/reference#chat-completions
const api_version = process.env['OPENAI_API_VERSION'] ?? '2023-07-01-preview'

const max_tokens = Number(process.env['OPENAI_MAX_TOKENS'] ?? 2000)
const temperature = Number(process.env['OPENAI_TEMPERATURE'] ?? 1)

log.debug({deployment, model, api_version, max_tokens, temperature})

// Configuration for Azure OpenAI
// https://github.com/openai/openai-node/issues/53#issuecomment-1517604780
const configuration = new Configuration({
    apiKey,
    basePath: `https://${deployment}.openai.azure.com/openai/deployments/${model}`,
    baseOptions: {
      headers: {'api-key': apiKey},
      params: {
        'api-version': api_version
      }
    }
})

const openai = new OpenAIApi(configuration)

const plugins: Map<string, PluginBase<any>> = new Map()
const functions: ChatCompletionFunctions[] = []

/**
 * Registers a plugin as a GPT function. These functions are sent to openAI when the user interacts with chatGPT.
 * @param plugin
 */
export function registerChatPlugin(plugin: PluginBase<any>) {
    plugins.set(plugin.key, plugin)
    functions.push({
        name: plugin.key,
        description: plugin.description,
        parameters: {
            type: 'object',
            properties: plugin.pluginArguments,
            required: plugin.requiredArguments
        }
    })
}

/**
 * Sends a message thread to chatGPT. The response can be the message responded by the AI model or the result of a
 * plugin call.
 * @param messages The message thread which should be sent.
 * @param msgData The message data of the last mattermost post representing the newest message in the message thread.
 */
export async function continueThread(messages: ChatCompletionRequestMessage[], msgData: MessageData): Promise<AiResponse> {
    let aiResponse: AiResponse = {
        message: 'Sorry, but it seems I found no valid response.'
    }

    // the number of rounds we're going to run at maximum
    let maxChainLength = 7;

    // check whether ChatGPT hallucinates a plugin name.
    const missingPlugins = new Set<string>()

    let isIntermediateResponse = true
    while(isIntermediateResponse && maxChainLength-- > 0) {
        const responseMessage = await createChatCompletion(messages, functions)
        log.trace(responseMessage)
        if(responseMessage) {
            // if the function_call is set, we have a plugin call
            if(responseMessage.function_call && responseMessage.function_call.name) {
                const pluginName = responseMessage.function_call.name;
                log.trace({pluginName})
                try {
                    const plugin = plugins.get(pluginName);
                    if (plugin){
                        const pluginArguments = JSON.parse(responseMessage.function_call.arguments ?? '[]');
                        log.trace({plugin, pluginArguments})
                        const pluginResponse = await plugin.runPlugin(pluginArguments, msgData)
                        log.trace({pluginResponse})

                        if(pluginResponse.intermediate) {
                            messages.push({
                                role: ChatCompletionResponseMessageRoleEnum.Function,
                                name: pluginName,
                                content: pluginResponse.message
                            })
                            continue
                        }
                        aiResponse = pluginResponse
                    } else {
                        if (!missingPlugins.has(pluginName)){
                            missingPlugins.add(pluginName)
                            log.debug({ error: 'Missing plugin ' + pluginName, pluginArguments: responseMessage.function_call.arguments})
                            messages.push({ role: 'system', content: `There is no plugin named '${pluginName}' available. Try without using that plugin.`})
                            continue
                        } else {
                            log.debug({ messages })
                            aiResponse.message = `Sorry, but it seems there was an error when using the plugin \`\`\`${pluginName}\`\`\`.`
                        }
                    }
                } catch (e) {
                    log.debug({ messages, error: e })
                    aiResponse.message = `Sorry, but it seems there was an error when using the plugin \`\`\`${pluginName}\`\`\`.`
                }
            } else if(responseMessage.content) {
                aiResponse.message = responseMessage.content
            }
        }

        isIntermediateResponse = false
    }

    return aiResponse
}

/**
 * Creates a openAI chat model response.
 * @param messages The message history the response is created for.
 * @param functions Function calls which can be called by the openAI model
 */
export async function createChatCompletion(messages: ChatCompletionRequestMessage[], functions: ChatCompletionFunctions[] | undefined = undefined): Promise<ChatCompletionResponseMessage | undefined> {
    const chatCompletionOptions: CreateChatCompletionRequest = {
        model: model,
        messages: messages,
        max_tokens: max_tokens,
        temperature: temperature,
    }
    if(functions) {
        chatCompletionOptions.functions = functions
        chatCompletionOptions.function_call = 'auto'
    }

    log.trace({chatCompletionOptions})

    const chatCompletion = await openai.createChatCompletion(chatCompletionOptions, proxyConfig)

    log.trace({chatCompletion})

    return chatCompletion.data?.choices?.[0]?.message
}

/**
 * Creates a openAI DALL-E response.
 * @param prompt The image description provided to DALL-E.
 */
export async function createImage(prompt: string): Promise<string | undefined> {
    const createImageOptions: CreateImageRequest = {
        prompt,
        n: 1,
        size: '512x512',
        response_format: 'b64_json'
    };
    log.trace({createImageOptions})
    const image = await openai.createImage(createImageOptions)
    log.trace({image})
    return image.data?.data[0]?.b64_json
}