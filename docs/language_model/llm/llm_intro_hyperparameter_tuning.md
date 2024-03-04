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
# Import the Ollama class from the llama_index.llms.ollama module.
from llama_index.llms.ollama import Ollama

# Create an instance of the Ollama class. The "gemma:7b" argument specifies the model to be used.
llm = Ollama(model="gemma:7b")

# Call the complete method on the Ollama instance. 
# The method generates a completion for the given prompt "What is Singapore?".
response = llm.complete("What is Singapore?")

# Print the generated response
print(response)
```

    /opt/conda/lib/python3.12/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


    Singapore is a city-state located on the island of Singapore. It is a Southeast Asian country and is known for its high standard of living, cleanliness, and efficiency.


### Question 2


```python
response = llm.complete("What is a Large Language Model?")
print(response)
```

    A Large Language Model (LLM) is a type of language model that has been trained on a massive amount of text data, typically billions or trillions of words. LLMs are designed to be able to understand and generate human-like text, engage in natural language processing tasks, and provide information and knowledge across a wide range of topics. LLMs are typically deep learning models that are trained using transformer architectures, such as the GPT-3 model.


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
# Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.01) make it more deterministic
temperature = 0.01
# Instantiate the Ollama class again
llm = Ollama(model="gemma:7b", temperature=temperature)

# Generate the response
response = llm.complete(prompt)

print(response)
```

    Sure, here's a happy birthday message for your friend:
    
    **Happy Birthday, [Friend's Name]!**
    
    I hope your day is filled with joy, laughter, and happiness. May all your wishes come true.
    
    Have a wonderful day, and I'm looking forward to celebrating with you soon.
    
    **Best regards,**
    
    [Your Name]



```python
# Check model
llm
```




    Ollama(callback_manager=<llama_index.core.callbacks.base.CallbackManager object at 0x7fd7e76754f0>, system_prompt=None, messages_to_prompt=<function messages_to_prompt at 0x7fd898237100>, completion_to_prompt=<function default_completion_to_prompt at 0x7fd8982bfd80>, output_parser=None, pydantic_program_mode=<PydanticProgramMode.DEFAULT: 'default'>, query_wrapper_prompt=None, base_url='http://localhost:11434', model='gemma:7b', temperature=0.01, context_window=3900, request_timeout=30.0, prompt_key='prompt', additional_kwargs={})




```python
# Run multiple times
for i in range(3):
    # Set the prompt
    prompt = "Write a happy birthday message, I would like to send to my friend."
    
    # Set the temperature
    # Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.01) make it more deterministic
    temperature = 0.01
    # Instantiate the Ollama class again
    llm = Ollama(model="gemma:7b", temperature=temperature)
    
    # Generate the response
    response = llm.complete(prompt)

    # Print 
    print('-'*10)
    print(f'Response {i}') 
    print('-'*10)
    print(response)

```

    ----------
    Response 0
    ----------
    Sure, here's a happy birthday message for your friend:
    
    **Happy Birthday, [Friend's Name]!**
    
    I hope your day is filled with joy, laughter, and happiness. May all your wishes come true.
    
    Have a wonderful day, and I'm looking forward to celebrating with you soon.
    
    **Best regards,**
    
    [Your Name]
    ----------
    Response 1
    ----------
    Sure, here's a happy birthday message for your friend:
    
    **Happy Birthday, [Friend's Name]!**
    
    I hope your day is filled with joy, laughter, and happiness. May all your wishes come true.
    
    Have a wonderful day, and I'm looking forward to celebrating with you soon.
    
    **Best regards,**
    
    [Your Name]
    ----------
    Response 2
    ----------
    Sure, here's a happy birthday message for your friend:
    
    **Happy Birthday, [Friend's Name]!**
    
    I hope your day is filled with joy, laughter, and happiness. May all your wishes come true.
    
    Have a wonderful day, and I'm looking forward to celebrating with you soon.
    
    **Best regards,**
    
    [Your Name]


**We can see above it is almost the exact same answer calling the LLM 3 times**.

#### High Temperature


```python
# Set the prompt
prompt = "Write a happy birthday message, I would like to send to my friend."

# Set the temperature
# Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.01) make it more deterministic
temperature = 1.0
# Instantiate the Ollama class again
llm = Ollama(model="gemma:7b", temperature=temperature)

# Generate the response
response = llm.complete(prompt)

print(response)

```

    Here are a few happy birthday messages you can send to your friend:
    
    **Short and sweet:**
    
    * "Happy Birthday, [friend's name]! Wishing you a day filled with joy!"
    * "Have a very happy birthday, [friend's name]! Can't wait to see you!"
    * "Happy Birthday, [friend's name]! May your day be filled with happiness!"
    
    **A little more personal:**
    
    * "Happy Birthday, [friend's name]! I hope your day is as special as you are."
    * "Have a wonderful birthday, [friend's name]! I'm so glad I have you in my life."
    * "Wishing you a very happy birthday, [friend's name]. Let's celebrate this special day together!"
    
    **Fun and cheeky:**
    
    * "Happy Birthday, [friend's name]! I hope your day is filled with cake and laughter."
    * "Have a great birthday, [friend's name]! I'm not going to tell you how old you are... for now, at least."
    * "Happy Birthday, [friend's name]! May your day be filled with all your favorite things... even if it's me."



```python
# Check model
llm
```




    Ollama(callback_manager=<llama_index.core.callbacks.base.CallbackManager object at 0x7fd7e8b82ea0>, system_prompt=None, messages_to_prompt=<function messages_to_prompt at 0x7fd898237100>, completion_to_prompt=<function default_completion_to_prompt at 0x7fd8982bfd80>, output_parser=None, pydantic_program_mode=<PydanticProgramMode.DEFAULT: 'default'>, query_wrapper_prompt=None, base_url='http://localhost:11434', model='gemma:7b', temperature=1.0, context_window=3900, request_timeout=30.0, prompt_key='prompt', additional_kwargs={})




```python
# Run multiple times
for i in range(3):
    # Set the prompt
    prompt = "Write a happy birthday message, I would like to send to my friend."
    
    # Set the temperature
    # Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.01) make it more deterministic
    temperature = 1.0
    # Instantiate the Ollama class again
    llm = Ollama(model="gemma:7b", temperature=temperature)
    
    # Generate the response
    response = llm.complete(prompt)

    # Print 
    print('-'*10)
    print(f'Response {i}') 
    print('-'*10)
    print(response)

```

    ----------
    Response 0
    ----------
    Here are a few happy birthday messages you can send to your friend:
    
    **Classic Wishes:**
    
    * "Happy Birthday, [Friend's Name]! Wishing you a day filled with joy, happiness, and laughter."
    * "Have a very happy birthday, [Friend's Name]! May your day be filled with sunshine and good times."
    * "Happy Birthday, my dear [Friend's Name]! I hope your day is as awesome as you are."
    
    **Personalized Wishes:**
    
    * "Happy Birthday, [Friend's Name]! I hope your day is filled with [specific things you know your friend enjoys]."
    * "I'm so glad it's your birthday, [Friend's Name]! I'm sending you a virtual hug and a bunch of birthday wishes."
    * "Wishing you a very happy birthday, [Friend's Name]! I can't wait to see what you have planned for this special day."
    
    **Fun and Quirky Wishes:**
    
    * "Happy Birthday, [Friend's Name]! May your day be filled with cake and laughter... and maybe a sprinkle of unicorn magic."
    * "Have a very happy birthday, [Friend's Name]! I'm hoping your day is as memorable as a trip to the moon."
    * "Happy Birthday, [Friend's Name]! I'm sending you virtual balloons and a party hat big enough for the both of us."
    ----------
    Response 1
    ----------
    Here are some happy birthday messages you can send to your friend:
    
    **Classic wishes:**
    
    * "Happy Birthday, [friend's name]! May your day be filled with joy, laughter, and good times."
    * "Have a very happy birthday, [friend's name]! I hope all your wishes come true."
    * "Wishing you a very happy birthday, [friend's name]! I'm sending you warmest wishes for a day filled with happiness."
    
    **Personalized wishes:**
    
    * "Happy Birthday, [friend's name]! I hope your day is as special as you are."
    * "Have a wonderful birthday, [friend's name]! I'm so glad to have you in my life."
    * "Sending you big birthday wishes, [friend's name]! I can't wait to see what you have planned."
    
    **Fun and cheesy:**
    
    * "Happy Birthday, [friend's name]! I'm hoping you have a day as awesome as you are."
    * "Have a blast on your birthday, [friend's name]! I'm planning on eating a cake in your honor."
    * "I'm not a party pooper, but I'm definitely not attending your party, [friend's name]. Have a great day!"
    
    **Remember:**
    
    * You can personalize the message with your friend's name and preferred gender-neutral pronouns.
    * You can add a specific wish or goal you have for your friend.
    * You can include a funny joke or a reference to a shared inside joke.
    * You can keep the message short and sweet, or you can write a longer, more heartfelt message.
    ----------
    Response 2
    ----------
    Sure, here's a happy birthday message you can send to your friend:
    
    **Happy Birthday, [Friend's Name]!**
    
    May your day be filled with joy, laughter, and happiness. I hope your special day is filled with all your favorite things, and I'm wishing you a very, very happy birthday!


**Here, you can see very varied answer in each of the 3 calls to the LLM commpared to lower temperature.**

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


## Summary

We covered the functionality of a basic LLM without any hyperparameter tuning. We then covered Temperature hyperparameter tuning. It is important to note that there are many hyperparameters that can be tuned, and we will update this tutorial to gradually include as many as we can.
