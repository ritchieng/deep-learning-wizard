# LLM Introduction & Hyperparameter Tuning

In this tutorial, we will be covering LLMs leveraging on Ollama and LlamaIndex using Gemma:7b (Google) open-source model.

## Environment Setup

Follow our [tutorial on Apptainer](https://www.deeplearningwizard.com/language_model/containers/hpc_containers_apptainer/) to get started. Once you have followed the tutorial and you completed the [Ollama, LlamaIndex and Gemma:7b](https://www.deeplearningwizard.com/language_model/containers/hpc_containers_apptainer/#ollama-gemma-workload section), you will be able to run `jupyter lab` in a new window to access and run this notebook.

!!! info  "Directory Guide"

    When you shell into the Apptainer `.sif` container, you will need to navigate the directory as you normally would into the Deep Learning Wizard repository that you cloned, requiring you to `cd ..` to go back a few directories and finally reaching the right folder. 

## Question and Answer | No Hyperparameter Tuning

In this section, we will leverage on the `Gemma:7b` LLM model to ask basic questions to get responses.

### Question 1


```python
# Import the Ollama class from the llama_index.llms module.
from llama_index.llms import Ollama

# Create an instance of the Ollama class. The "gemma:7b" argument specifies the model to be used.
llm = Ollama(model="gemma:7b")

# Call the complete method on the Ollama instance. 
# The method generates a completion for the given prompt "What is Singapore?".
response = llm.complete("What is Singapore?")

# Print the generated response
print(response)
```

    Singapore is a city-state located in Southeast Asia. It is a wealthy and cosmopolitan city known for its modern architecture, efficient public transportation system, and world-class shopping malls.


### Question 2


```python
response = llm.complete("What is a Large Language Model?")
print(response)
```

    **Large Language Models (LLMs)** are a type of language model that are trained on massive amounts of text data, typically billions or even trillions of words. LLMs are designed to understand and generate human-like text, and they are often used for a wide range of tasks, including:
    
    * **Text summarization:** Converting large amounts of text into shorter summaries.
    * **Text generation:** Generating new text, such as articles, stories, or code.
    * **Translation:** Translating text from one language to another.
    * **Question answering:** Answering questions based on text.
    * **Code generation:** Generating code in various programming languages.
    
    **Key Characteristics of LLMs:**
    
    * **Large-scale training:** LLMs are trained on massive datasets, typically billions or trillions of words.
    * **Text-centric:** LLMs are primarily designed to understand and generate text.
    * **Multi-task learning:** LLMs can be trained to perform multiple tasks, such as text summarization, translation, and code generation.
    * **Transfer learning:** LLMs can be fine-tuned for specific tasks, leveraging their general language understanding abilities.
    * **Human-like text generation:** LLMs can generate text that resembles human writing, often with high accuracy.
    
    **Examples of LLMs:**
    
    * **GPT-3:** A popular LLM that is known for its ability to generate high-quality text.
    * **GPT-Neo:** A variant of GPT-3 that is designed for text generation tasks.
    * **Transformer-XL:** A large language model that is based on the transformer architecture.
    
    **Applications:**
    
    LLMs have a wide range of potential applications, including:
    
    * Text and code editing
    * Content creation
    * Information retrieval
    * Machine translation
    * Code development
    * Customer service
    
    **Note:** LLMs are still under development, and their capabilities are constantly evolving.


In our second question, we change the question to "What is a Large Language Model?" and you can observe how the answer is substantially longer than the first question "What is Singapore". In the next section, you will discover that this relates to a few hyperparemeters in LLMs that can be tweaked.

## Question and Answer | Hyperparameter Tuning

### Temperature Tuning

The `temperature` parameter in LLMs plays a pivotal role in determining the predictability of the output. 

- **Lower temperature values (e.g., 0.2)** lead to more predictable and consistent responses, but may risk being overly constrained or repetitive.

- **Higher temperature values (e.g., 1.0)** introduce more randomness and diversity, but can result in less consistency and occasional incoherence.
The choice of temperature value is a trade-off between consistency and variety, and should be tailored to the specific requirements of your task.

#### Low Temperature


```python
# Set the prompt
prompt = "Write a happy birthday message, I would like to send to my friend."

# Set the temperature
# Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.2) make it more deterministic
temperature = 0.6

# Generate the response
response = llm.complete(prompt, temperature=temperature)

print(response)

```

    Happy Birthday, [Friend's name]! I hope your day is filled with joy, laughter, and happiness! 🎉🎂


#### High Temperature


```python
# Set the prompt
prompt = "Write a happy birthday message, I would like to send to my friend."

# Set the temperature
# Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.2) make it more deterministic
temperature = 1.0

# Generate the response
response = llm.complete(prompt, temperature=temperature)

print(response)

```

    Sure, here are some happy birthday messages you can send to your friend:
    
    **Simple and sweet:**
    
    * "Happy Birthday, [friend's name]! Wishing you a day filled with joy and happiness."
    * "Have a very happy birthday, [friend's name]! Sending positive vibes your way."
    * "Wishing you a happy, happy birthday, [friend's name]! I hope it's a day full of treats."
    
    **Funny:**
    
    * "Happy Birthday, [friend's name]! I'm not going to reveal how old you are, but I will say you're definitely not a teenager."
    * "Hope your birthday is as awesome as you are, [friend's name]. I'm not kidding."
    * "Have a birthday as bright as your smile, [friend's name]. I'm not a very creative person, but I tried."
    
    **Personal:**
    
    * "Happy Birthday, [friend's name]! I'm so glad to have you in my life."
    * "I'm wishing you a very happy birthday, [friend's name]. I hope your day is filled with all your favorite things."
    * "I know it's your birthday today, [friend's name]. I'm sending you lots of love and best wishes."


#### Mathematical Interpretation

In LLMs, the `temperature` parameter is used to control the randomness of predictions by scaling the logits before applying soft(arg)max.

- The model computes a score (also known as logits) for each possible next token based on the current context. These scores are then transformed into probabilities using the soft(arg)max function.

- The soft(arg)max function is defined as follows:

    $$\text{softargmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

- Before applying soft(arg)max, the logits are divided by the `temperature` value. This process is called temperature scaling and the equation becomes:

    $$\text{softargmax}(x_i) = \frac{e^{x_i/T}}{\sum_j e^{x_j/T}}$$

- When `T` > 1, it makes the distribution more uniform (increases randomness). When `T` < 1, it makes the distribution more peaky (reduces randomness).

So, in simple terms, a higher temperature value makes the model's output more random, and a lower temperature makes it more deterministic.


```python
import numpy as np

def softargmax(x, T=1.0):
    e_x = np.exp(x / T)
    return e_x / e_x.sum()

# Define logits
logits = np.array([0.2, 0.3, 0.1, 0.4])

# Compute soft(arg)max for different temperatures
for T in [0.1, 1.0]:
    print(f"Temperature: {T}")
    print(softargmax(logits, T=T))
    print()
```

    Temperature: 0.1
    [0.08714432 0.23688282 0.0320586  0.64391426]
    
    Temperature: 1.0
    [0.23632778 0.26118259 0.21383822 0.28865141]
    


In the Python code above leveraging on `numpy` library, you can see that

- `softargmax` is a function that computes the soft(arg)max of an array of logits `x` for a given temperature `T`.
- We define an array of logits and compute the soft(arg)max for different temperatures.
- When you run this code, you'll see that as the temperature increases, the soft(arg)max output becomes more uniform (i.e., the probabilities are more evenly distributed), and as the temperature decreases, the soft(arg)max output becomes more peaky (i.e., one probability dominates the others). This illustrates how temperature can control the randomness of the model's output.

- To close this off, taking the max of the soft(arg)max output, you will observe how it gets more random in the max value as the soft(arg)max output becomes more uniform. This links to the concept of how the next word gets more random because of the max of the uniformity of the soft(arg)max output.

### Top-K Tuning

In LLMs, the `top_k` hyperparameter is a key factor that influences the unpredictability of the generated output.
- **For smaller `top_k` values**: The model behaves in a more predictable manner. It only takes into account a limited set of the most probable next tokens at each step of the generation process. This can result in responses that are more concise and consistent, but there’s a possibility that the output may be too restricted or repetitive.
- **For larger `top_k` values**: The model takes into consideration a broader set of potential next tokens. This infuses more variety and randomness into the generated output. However, the responses can become less consistent and may occasionally be less coherent or pertinent.
Therefore, the selection of the top_k value can be viewed as a balance between consistency and variety in the model’s responses. It’s crucial to adjust this parameter based on the specific needs of your task. 

#### Low K


```python
# Set the prompt
prompt = "Write a happy birthday message, I would like to send to my friend."

# Set the temperature
# Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.2) make it more deterministic
temperature = 0.8

# Set the top_k
# This parameter controls the number of tokens considered for each step of the generation process
top_k = 50

# Generate the response
response = llm.complete(prompt, temperature=temperature, top_k=top_k)

print(response)

```

    Happy Birthday, [Friend's Name]! I hope your day is filled with joy, laughter, and happiness. May all your wishes come true! 🎉🎂


#### High K


```python
# Set the prompt
prompt = "Write a happy birthday message, I would like to send to my friend."

# Set the temperature
# Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.2) make it more deterministic
temperature = 0.8

# Set the top_k
# This parameter controls the number of tokens considered for each step of the generation process
top_k = 100

# Generate the response
response = llm.complete(prompt, temperature=temperature, top_k=top_k)

print(response)

```

    Sure, here are a few options:
    
    **Short and sweet:**
    
    * "Happy Birthday, [friend's name]! Wishing you a day filled with joy and happiness."
    * "Have a very happy birthday, [friend's name]! Let's celebrate!"
    
    **More personal:**
    
    * "Happy Birthday, [friend's name]! I hope your day is as special as you are."
    * "Wishing you a very happy birthday, [friend's name]. I'm so glad to have you in my life."
    
    **Funny:**
    
    * "Happy Birthday, [friend's name]! I'm not gonna lie, I'm a little jealous of your awesome day."
    * "Have a truly happy birthday, [friend's name]. I'm not sure if you're actually going to be happy, but I'm hoping for your sake."


You can observe that the reply is more diverse with a high `top_k` hyperparameter.

#### Mathematical Interpretation

In LLMs, the `top_k` parameter is used to limit the number of next tokens considered for generation.

- After computing the soft(arg)max probabilities for all possible next tokens, the model sorts these probabilities in descending order.

- The model then only considers the `top_k` tokens with the highest probabilities for the next step of the generation process.

- This process is called `top_k` sampling.

Here's a simple Python code snippet that illustrates how `top_k` works.


```python
def top_k(logits, k):
    # Sort the logits
    sorted_indices = np.argsort(logits)
    
    # Consider only the top k
    top_k_indices = sorted_indices[-k:]
    
    # Create a new array with only the top k probabilities
    top_k_logits = logits[top_k_indices]
    
    return top_k_logits

# Define logits
logits = np.array([0.2, 0.3, 0.1, 0.4])

# Compute top_k for different values of k
for k in [2, 3, 4]:
    print(f"Top {k} logits:")
    print(top_k(logits, k=k))
    print()
```

    Top 2 logits:
    [0.3 0.4]
    
    Top 3 logits:
    [0.2 0.3 0.4]
    
    Top 4 logits:
    [0.1 0.2 0.3 0.4]
    


In the code above

- `top_k` is a function that computes the top `k` logits from an array of logits.

- We define an array of logits and compute the top `k` logits for different values of `k`.

- When you run this code, you'll see that as `k` increases, more logits are considered. This illustrates how `top_k` can control the number of tokens considered by the model.

## Summary

We covered the functionality of a basic LLM without any hyperparameter tuning. We then covered Temperature and Top-K hyperparameter tuning. It is important to note that there are many hyperparameters that can be tuned, and we will update this tutorial to gradually include as many as we can.
