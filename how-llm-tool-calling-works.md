# When an LLM calls a tool, who actually waits?

If a language model takes tokens as input and predicts the next token, where does a function call fit? Take a weather app: at what point does predicting text about the weather turn into querying a forecast service?

It happens at the boundary between **two loops**. The inference engine's inner loop generates tokens that *describe* a call. The application's outer loop interprets the completed message, executes the function, and adds its result to the input for the next model request. Let's follow one call across that boundary.

## One user question, two model requests

Suppose the app exposes `get_forecast(area: string)`. It sends the tool's name, description, and argument schema alongside the user's question. The implementation stays in the app. A tool-trained model has seen examples where the appropriate continuation is a tool-call record: it can produce that format using the same next-token mechanism it uses to produce prose.

Here's an illustrative trace for an application-managed tool. The forecast is made up; the message fields are abbreviated.

```text
Request 1: app → inference service
  user: I'm walking from Tanjong Pagar MRT to Maxwell Food Centre for lunch. Should I take an umbrella?
  tools: get_forecast(area: string)

Response 1: inference service → app
  assistant.tool_calls: [{id: c1, name: get_forecast, arguments: {area: "Tanjong Pagar"}}]

Outside inference: app → weather service → app
  get_forecast("Tanjong Pagar") returns {period: "lunchtime", rain_probability: 0.8}

Request 2: app → inference service
  original messages + assistant's call c1
  + tool result for c1: {period: "lunchtime", rain_probability: 0.8}

Response 2: inference service → app
  assistant: I'd take one. The forecast gives an 80% chance of rain at lunchtime.
```

**Response 1 really returns.** The inference service recognizes the model's call format and exposes it to the application (for example, as `tool_calls`). With streaming, the arguments may arrive in pieces. The application waits for a complete call, checks the tool name, permissions, and arguments, then dispatches it. The model hasn't opened a connection to the weather service. Its first generation is over.

**Request 2 is a new inference request.** The tool result is serialized as input, tokenized, and embedded like the rest of the conversation. The call ID pairs the result with the right request, which matters if the assistant asked for two neighbourhoods at once. The model can now use the forecast because it is in the context; no weights changed.

## Inside the first request: how do tokens become a valid argument?

At each decode step the Transformer produces a hidden vector for the last position. Its output layer maps that vector to **logits**: one raw score per vocabulary token. The vocabulary is already the list of candidates; index `i` is the score for token ID `i`. A decoding policy turns those scores into a choice.

For a small worked example, let the weather tool also accept a boolean `hourly` argument. The model has generated `{"hourly":` and must supply a value. Pretend each continuation below is one token and these are the only candidates. If the serving mode enforces the schema with a grammar, this is what happens at temperature 1, without top-k or top-p filtering:

| Next text | Model logit | After grammar mask | Sampling probability |
|---|---:|---:|---:|
| `"yes"` (a string) | 3 | −∞ | 0% |
| `true` | 2 | 2 | 73.1% |
| `false` | 1 | 1 | 26.9% |

The model preferred `"yes"`, but a string cannot fill a boolean field. Its masked score contributes zero to the softmax. Among the two remaining tokens, `P(true) = exp(2) / (exp(2) + exp(1)) ≈ 0.731`. Greedy decoding would simply take the higher legal score; sampling can still pick `false`.

```python
# Inner loop: one completion, simplified. grammar is optional, per response.
while not stopped(generated):
    logits = model.next_token_logits(prompt_ids + generated)
    if grammar is not None:
        mask_illegal_in_place(logits, grammar)  # illegal entries become -inf
    if config.greedy:
        token = argmax(logits)
    else:
        p = softmax(logits / config.temperature)  # T > 0
        token = sample(filter_and_renormalize(p, config.top_k, config.top_p))
    generated.append(token)
    if grammar is not None:
        grammar.advance(token)
```

Why advance the grammar? Before `true`, the decoder needed a value. After it, a closing `}` might be legal. Real tokens can cover part of a word or multiple characters, so the allowed set depends on the exact prefix. The grammar can enforce supported structural rules; it cannot tell whether the caller *wants* hourly data, whether the forecast covers the intended neighbourhood, or whether the call is authorized. Tool calling doesn't inherently require constrained decoding, either: some serving modes parse the generated call afterward.

## Outside the model: the two awaits

```python
# Outer loop: application-side pseudocode
for _ in range(max_model_requests):
    reply = await infer(history, tool_specs)       # (1) wait for generation
    if reply.incomplete:
        return error("Incomplete model response")
    if not reply.tool_calls:
        return reply.text
    history.append(reply.assistant_message)
    for call in reply.tool_calls:
        result = await execute_checked(call)       # (2) wait for the tool
        history.append(tool_message(call.id, result))
return error("Tool-call budget exhausted")
```

By await **(2)**, await **(1)** has already returned. That distinction survives an SDK wrapping the whole conversation in a single `run()`. In a longer chain, `resolve_place("Maxwell Food Centre")` might have to finish before `get_forecast(place_id)` can start. Independent calls can instead be dispatched concurrently and their results paired by ID. A timeout or a tool error is something the application handles, rather than a reason to leave a decode loop spinning.

**What to measure:** put separate spans around model request 1, tool execution, and model request 2. If the weather service takes a second, a user can see a second of delay without a second of active GPU decode for that request. Conversely, a fast tool that returns a huge JSON payload can make the *next* model request slow: the newly appended tokens still need prefill, even when the old prefix hits the KV cache. Those two traces call for different fixes.

For the book treatment and the surrounding serving-cost model, see [Appendix H §H.1.4 of *LLM Inference Deployment and Acceleration*](https://github.com/ZiluTian/ai-infra-prep/blob/main/appendix/附录H-Agent负载与自我改进.md).
