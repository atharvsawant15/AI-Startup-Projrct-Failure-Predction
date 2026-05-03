# generate explanation and recommendations using LLM with environment variable support

import os
from dotenv import load_dotenv
from openai import OpenAI


# load environment variables from .env file
load_dotenv()

# initialize OpenAI client using API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_explanation(data, prediction, prob_failure):
    # determine outcome label
    outcome = "FAILURE" if prediction == 1 else "SUCCESS"

    # create prompt for LLM
    prompt = f"""
You are an expert startup analyst.

Startup Data:
{data}

Prediction:
{outcome}

Failure Probability: {round(prob_failure, 3)}

Tasks:
1. Explain in simple terms why this prediction occurred
2. Identify key risk or success factors
3. If FAILURE:
   - Give 5 actionable recommendations to improve success chances
4. If SUCCESS:
   - Give 5 precautions to maintain growth and avoid failure

Keep response structured with:
- Explanation
- Key Factors
- Recommendations
"""

    # call OpenAI API
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful startup advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    # return generated explanation
    return response.choices[0].message.content