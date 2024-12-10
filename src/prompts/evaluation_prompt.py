EVALUATION_PROMPT = """
{{
  "objective": "You are given a detailed technical document and a component name in mechanical and automotive systems domain. Your task is to check whether the given component name is relevant to the document and explain why.",
  "instructions": [
      "Step 1: Read and thoroughly understand the provided technical document.",
      "Step 2: Familiarize yourself with the concept of the given component as described in the technical documentation.",
      "Step 3: Verify if the given component name is relevant to the keyword phrase or related concepts within the technical document."
  ],
  "note":[
      "Return output as the format in evaluation section, including component, judgent, reasoning in json format"
  ],
  "input": {{
    "document": "{document}",
    "component": "{component}"
  }},
  "evaluation": {{
    "component": "{component}"
    "filename": "{filename}"
    "judgment": "[Relevant| Part Relevant|Not Relevant]",
    "reasoning": "[Explain the reason]"
  }}
}}
"""