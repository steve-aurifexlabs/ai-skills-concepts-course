import 'process'
import { Configuration, OpenAIApi } from 'openai'

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(configuration);

var prompt = `
Write a PyTorch training function that predicts demand as a function of time of day and temperature.

Load data from a CSV file './data/demand_temperature.csv'.
Here are the first 3 lines:
Time,Demand,Temperature
2014-01-01 00:00:00,3.794,18.05
2014-01-01 01:00:00,3.418,17.2
  
Convert Time into time of day before using as an input.
Create and use a data loader with 10% of the data for eval.

Use SGD and CrossEntropyLoss.
Do not include batches.
Be sure to convert data to tensors and use cuda if possible.
The model should use nn.Sequential, have one hidden layer with 10 neurons, and ReLU activations.

Make the file a script that calls the train function.

Make the code as simple as possible. The code should work. All types should be correct.
`



console.log('Prompt:', prompt)

const completion = await openai.createChatCompletion({
  model: "gpt-3.5-turbo",
  messages: [
    {"role": "system", "content": "You are an expert Python developer and instructor. You include lots of comments. You use readable variable names. You always format the output as a single python script with any prose as docstring comments in the python code. You omit the ```python and ``` start and end tokens."},
    {role: "user", content: prompt}
  ],
  temperature: 0.2,
})

console.log(completion.data.choices[0].message.content.toString())
