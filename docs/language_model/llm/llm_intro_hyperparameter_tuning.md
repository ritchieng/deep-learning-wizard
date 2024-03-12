---
comments: true
---

# LLM Introduction & Hyperparameter Tuning

In this tutorial, we will be covering LLMs leveraging on Ollama using Gemma:7b (Google) open-source model.

## Environment Setup

Follow our [tutorial on Apptainer](https://www.deeplearningwizard.com/language_model/containers/hpc_containers_apptainer/) to get started. Once you have followed the tutorial till the [Ollama section](https://www.deeplearningwizard.com/language_model/containers/hpc_containers_apptainer/#ollama-gemma-workloads) where you successfully ran `ollama serve` and `ollama run gemma:7b`, you can run the `apptainer shell --nv --nvccli apptainer_container_0.1.sif` command followed by `jupyter lab` to access and run this notebook.

!!! info  "Directory Guide"

    When you shell into the Apptainer `.sif` container, you will need to navigate the directory as you normally would into the Deep Learning Wizard repository that you cloned, requiring you to `cd ..` to go back a few directories and finally reaching the right folder. 

## Question and Answer | No Hyperparameter Tuning

In this section, we will leverage on the `Gemma:7b` LLM model to ask basic questions to get responses.


```python
import ollama
```

### Question 1


```python
# The 'chat' function is called with two parameters: 'model' and 'messages'.
response = ollama.chat(
    model='gemma:7b',  # The 'model' parameter specifies the model to be used. Here, 'gemma:7b' is the model.
    messages=[  # The 'messages' parameter is a list of message objects.
        {
            'role': 'user',  # Each message object has a 'role' key. It can be 'user' or 'assistant'.
            'content': 'What is Singapore?',  # The 'content' key contains the actual content of the message.
        },
    ]
)

# The 'chat' function returns a response object. 
# The content of the assistant's message is accessed using the keys 'message' and 'content'.
# The 'print' function is used to display this content.
print(response['message']['content'])
```

    Singapore is a city-state located on the island of Singapore, a Southeast Asian island. It is a highly developed city known for its modern architecture, efficient transportation system, and vibrant cultural diversity.


### Question 2


```python
response = ollama.chat(
    model='gemma:7b',
    messages=[
        {
            'role': 'user',
            'content': 'What is a Large Language Model?',
        },
    ]
)

print(response['message']['content'])
```

    Sure, a Large Language Model (LLM) is a type of language model that has been trained on a massive amount of text data and has the ability to engage in a wide range of natural language processing tasks. LLMs are typically designed to have a large number of parameters, which allows them to learn complex relationships between words and sentences. LLMs are often used for tasks such as text summarization, translation, and code generation.


In our second question, we change the question to "What is a Large Language Model?" and you can observe how the answer is slightly longer than the first question "What is Singapore". In the next section, you will discover that this relates to a few hyperparemeters in LLMs that can be tweaked.

## Question and Answer | Hyperparameter Tuning

### Temperature Tuning

The `temperature` parameter in LLMs plays a pivotal role in determining the predictability of the output. 

- **Lower temperature values (e.g., 0.2)** lead to more predictable and consistent responses, but may risk being overly constrained or repetitive.

- **Higher temperature values (e.g., 1.0)** introduce more randomness and diversity, but can result in less consistency and occasional incoherence.
The choice of temperature value is a trade-off between consistency and variety, and should be tailored to the specific requirements of your task.

#### Low Temperature


```python
# Create new model
# Set the temperature
# Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.1) make it more deterministic
modelfile='''
FROM gemma:7b
PARAMETER temperature 0.1
'''
ollama.create('gemma_low_temp', modelfile=modelfile)

# Now you can use the new model with adjusted temperature
response = ollama.chat(
    model='gemma_low_temp',
    messages=[
        {
            'role': 'user',
            'content': 'Write a happy birthday message, I would like to send to my friend.',
        },
    ]
)

print(response['message']['content'])
```

    Happy Birthday, [Friend's Name]! I hope your day is filled with joy, laughter, and happiness. May all your wishes come true! 🎉🎂



```python
# Run multiple times
for i in range(3):
    response = ollama.chat(
        model='gemma_low_temp',
        messages=[
            {
                'role': 'user',
                'content': 'Write a happy birthday message, I would like to send to my friend.',
            },
        ],
    )
    
    # Print 
    print('-'*10)
    print(f'Response {i}') 
    print('-'*10)
    print(response['message']['content'])
```

    ----------
    Response 0
    ----------
    Happy Birthday, [Friend's Name]! I hope your day is filled with joy, laughter, and happiness. May all your wishes come true! 🎉🎂
    ----------
    Response 1
    ----------
    Happy Birthday, [Friend's Name]! I hope your day is filled with joy, laughter, and happiness. May all your wishes come true! 🎉🎂
    ----------
    Response 2
    ----------
    Happy Birthday, [Friend's Name]! I hope your day is filled with joy, laughter, and happiness. May all your wishes come true! 🎉🎂


**We can see above it is the exact same answer calling the LLM 3 times**.

#### High Temperature


```python
# Create new model
# Set the temperature
# Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.1) make it more deterministic
modelfile='''
FROM gemma:7b
PARAMETER temperature 1.0
'''
ollama.create('gemma_high_temp', modelfile=modelfile)

# Now you can use the new model with adjusted temperature
response = ollama.chat(
    model='gemma_high_temp',
    messages=[
        {
            'role': 'user',
            'content': 'Write a happy birthday message, I would like to send to my friend.',
        },
    ]
)

print(response['message']['content'])
```

    Happy Birthday, [Friend's Name]! I hope your day is filled with joy, laughter, and happiness! 🎉🎂 🎉



```python
# Run multiple times
for i in range(3):
    response = ollama.chat(
        model='gemma_high_temp',
        messages=[
            {
                'role': 'user',
                'content': 'Write a happy birthday message, I would like to send to my friend.',
            },
        ],
    )
    
    # Print 
    print('-'*10)
    print(f'Response {i}') 
    print('-'*10)
    print(response['message']['content'])
```

    ----------
    Response 0
    ----------
    Happy Birthday, [Friend's Name]! I hope your day is filled with joy, laughter, and happiness. Let's celebrate your special day together!
    ----------
    Response 1
    ----------
    Happy Birthday, [Friend's Name]! I hope your day is filled with joy, laughter, and happiness. Wishing you a very special day filled with memorable moments and sweet treats!
    ----------
    Response 2
    ----------
    Sure, here is a happy birthday message you can send to your friend:
    
    **Happy Birthday, [Friend's Name]!**
    
    May your day be filled with joy, laughter, and happiness. I hope your special day is filled with everything you wish for. I'm sending you positive vibes and can't wait to see you soon.


**Here, you can see very varied answer in each of the 3 calls to the LLM compared to lower temperature.**

#### Mathematical Interpretation

In LLMs, the `temperature` parameter is used to control the randomness of predictions by scaling the logits before applying soft(arg)max.

- The model computes a score (also known as logits) for each possible next token based on the current context. These scores are then transformed into probabilities using the soft(arg)max function.

- The soft(arg)max function is defined as follows:

    $$\text{softargmax}(x_i) = \frac{e^{x_i}}{\sum_i^j e^{x_i}}$$

- Before applying soft(arg)max, the logits are divided by the `temperature` value. This process is called temperature scaling and the equation becomes:

    $$\text{softargmax}(x_i) = \frac{e^{x_i/T}}{\sum_i^j e^{x_i/T}}$$

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
# Create new model

# Set the temperature
# Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.1) make it more deterministic

# Set the top_k
# This parameter controls the number of tokens considered for each step of the generation process
modelfile='''
FROM gemma:7b
PARAMETER temperature 0.5
PARAMETER top_k 3
'''
ollama.create('gemma_topk_3', modelfile=modelfile)

# Now you can use the new model with adjusted temperature
response = ollama.chat(
    model='gemma_topk_3',
    messages=[
        {
            'role': 'user',
            'content': 'Write a happy birthday message, I would like to send to my friend.',
        },
    ]
)

print(response['message']['content'])
```

    Happy Birthday, [Friend's Name]! I hope your day is filled with joy, laughter, and happiness. May all your wishes come true!



```python
# Run multiple times
for i in range(3):
    response = ollama.chat(
        model='gemma_topk_3',
        messages=[
            {
                'role': 'user',
                'content': 'Write a happy birthday message, I would like to send to my friend.',
            },
        ],
    )
    
    # Print 
    print('-'*10)
    print(f'Response {i}') 
    print('-'*10)
    print(response['message']['content'])
```

    ----------
    Response 0
    ----------
    Sure, here's a happy birthday message for your friend:
    
    **Happy Birthday, [Friend's Name]!**
    
    I hope your day is filled with joy, laughter, and happiness. May all your wishes come true. Have a blast! 🎉🎂🎉
    ----------
    Response 1
    ----------
    Happy Birthday, [Friend's Name]! I hope your day is filled with joy, laughter, and happiness. May all your wishes come true. 🎉🎂
    ----------
    Response 2
    ----------
    Happy Birthday, [Friend's Name]! I hope your day is filled with joy, laughter, and happiness. May all your wishes come true. 🎉🎂


#### High K


```python
# Create new model

# Set the temperature
# Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.1) make it more deterministic

# Set the top_k
# This parameter controls the number of tokens considered for each step of the generation process
modelfile='''
FROM gemma:7b
PARAMETER temperature 0.5
PARAMETER top_k 200
'''
ollama.create('gemma_topk_200', modelfile=modelfile)

# Now you can use the new model with adjusted temperature
response = ollama.chat(
    model='gemma_topk_200',
    messages=[
        {
            'role': 'user',
            'content': 'Write a happy birthday message, I would like to send to my friend.',
        },
    ]
)

print(response['message']['content'])
```

    Happy Birthday, [Friend's name]! I hope your day is filled with joy, laughter, and happiness! 🎉🎂



```python
# Run multiple times
for i in range(3):
    response = ollama.chat(
        model='gemma_topk_200',
        messages=[
            {
                'role': 'user',
                'content': 'Write a happy birthday message, I would like to send to my friend.',
            },
        ],
    )
    
    # Print 
    print('-'*10)
    print(f'Response {i}') 
    print('-'*10)
    print(response['message']['content'])
```

    ----------
    Response 0
    ----------
    Sure, here is a happy birthday message for your friend:
    
    **Happy Birthday, [Friend's Name]!**
    
    I hope your day is filled with joy, laughter, and happiness. May all your wishes come true today.
    
    Have a wonderful birthday, and I look forward to seeing you soon.
    
    **Best regards,**
    
    [Your Name]
    ----------
    Response 1
    ----------
    Sure, here's a happy birthday message for your friend:
    
    **Happy Birthday, [Friend's Name]!**
    
    May your day be filled with joy, laughter, and happiness. I hope your special day is filled with all your favorite things and that your wishes come true.
    
    **Have a wonderful birthday, my dear friend!**
    ----------
    Response 2
    ----------
    Sure, here's a happy birthday message for your friend:
    
    **Happy Birthday, [Friend's Name]!**
    
    May your day be filled with joy, laughter, and happiness. I hope you have a wonderful time celebrating your special day!


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
