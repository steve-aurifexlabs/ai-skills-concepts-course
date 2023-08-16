# select prompt
promptIndex = 0

# imports
import os
import json
from string import Template
import pandas as pd
import openai

# openai credentials
openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")

# load prompt
with open('prompts.json') as promptsFile:
    prompts = json.load(promptsFile)

prompt = prompts[promptIndex]

# load csv file
dataFrame = pd.read_csv(os.path.join('data', 'classifier_data_0.csv'))

# create random batch
batch = dataFrame.sample(n=100)
print(batch)


# fetch live data (optional)

# create prompt from template - TODO
finalPrompt = prompt

# generate output
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": finalPrompt}
  ]
)

print(completion.choices[0].message)

# log prompt and output