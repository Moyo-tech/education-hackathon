import { Configuration, OpenAIApi } from "openai";
import dotenv from "dotenv";

dotenv.config();

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openai = new OpenAIApi(configuration);

export async function runQuery(query) {
  const modifiedQuery =
    "List me 5 career paths that I can take in " +
    query +
    ". Only five lists and only names of the paths.";

  const queryParams = await openai.createCompletion({
    model: "text-davinci-003",
    prompt: modifiedQuery,
  });

  return queryParams.data.choices[0].text;
}