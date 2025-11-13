# coding=utf-8  
# Copyright 2025 The Google Research Authors.  
#  
# Licensed under the Apache License, Version 2.0 (the "License");  
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at  
#  
#     http://www.apache.org/licenses/LICENSE-2.0  
#  
# Unless required by applicable law or agreed to in writing, software  
# distributed under the License is distributed on an "AS IS" BASIS,  
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and  
# limitations under the License.  
  
pyreact_solve_table_prompt = '''  
You are working with a pandas dataframe regarding "{table_caption}" in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}  
  
Tool description:  
- `python_repl_ast`: A Python shell. Use this to execute python commands. Input should be a valid one-line python command.  
  
Strictly follow the given format to respond:  
Thought: you should always think about what to do  
Action: the Python command to execute  
Observation: the result of the action  
... (this Thought/Action/Observation can repeat N times)  
Thought: before giving the final answer, you should think about the observations  
Final Answer: the final answer to the original input question (numerical  or string)  
  
Notes:  
- Do not use markdown or any other formatting in your responses.  
- Ensure the last line is  "Final Answer: [numerical (e.g., 78.1) or string (e.g., women)]" form.  
- Directly output the Final Answer rather than outputting by Python.  
- Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.  
- Pay special attention to multi-level column headers and ensure you're accessing the correct data.  
  
You are working with the following table regarding "{table_caption}":  
{table}  
  
Please answer the question: {query}.  
  
Begin!  
'''  
  
tablerag_extract_column_prompt = '''  
Given a large table regarding "{table_caption}", I want to answer a question: {query}  
Since I cannot view the table directly, please suggest some column names that might contain the necessary data to answer this question.  
Please answer with a list of column names in JSON format without any additional explanation.  
Example:  
["column1", "column2", "column3"]  
'''  
  
tablerag_extract_cell_prompt = '''  
Given a large table regarding "{table_caption}", I want to answer a question: {query}  
Please extract some keywords which might appear in the table cells and help answer the question.  
The keywords should be categorical values rather than numerical values.  
The keywords should be contained in the question and should not be a column name.  
Please answer with a list of keywords in JSON format without any additional explanation.  
Example:  
["keyword1", "keyword2", "keyword3"]  
'''  
  
# tablerag_solve_table_prompt = '''  
# You are working with a pandas dataframe regarding "{table_caption}" in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}  
  
# Tool description:  
# - `python_repl_ast`: A Python interactive shell. Use this to execute python commands. Input should be a valid single line python command.  
  
# Since you cannot view the table directly, here are some schemas and cell values retrieved from the table.  
  
# {schema_retrieval_result}  
  
# {cell_retrieval_result}  
  
# Strictly follow the given format to respond:  
# Thought: you should always think about what to do  
# Action: the single line Python command to execute  
# Observation: the result of the action  
# ... (this Thought/Action/Observation can repeat N times)  
# Thought: before giving the final answer, you should think about the observations  
# Final Answer: the final answer to the original input question (numerical value or string)  
  
# Notes:  
# - Do not use markdown or any other formatting in your responses.  
# - The question's "proportion" = the table's "percent" value (no division needed).
# - Ensure the last line is  "Final Answer: [numerical value or string]" form.
# - Directly output the Final Answer rather than outputting by Python.  
# - Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.  
# - Pay special attention to multi-level column headers and ensure you're accessing the correct data.  
  
# Now, given a table regarding "{table_caption}", please use `python_repl_ast` with the column names and cell values above to answer the question: {query}  
  
# Begin!  
# '''

# tablerag_solve_table_prompt = '''  
# You are working with a pandas dataframe regarding "{table_caption}" in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}  

# Tool description:  
# - `python_repl_ast`: A Python interactive shell. Use this to execute python commands. Input must be a valid single-line python command (no natural language).  

# Critical Context:  
# You cannot view the full table, but {schema_retrieval_result} and {cell_retrieval_result} provide column schemas (e.g., multi-level headers) and key cell values. Use these to locate data.  

# ### Universal Reasoning Flow (MUST FOLLOW IN ORDER)  
# Step 1: Parse column schema to confirm data location  
# Thought: Analyze {schema_retrieval_result} to identify target columns (use tuple for multi-level headers, e.g., ("gender", "education")).  
# Action: print("Target columns to confirm:", df.columns.tolist())  

# Step 2: Locate rows with key cell values  
# Thought: Match {cell_retrieval_result} with dataframe rows to find the category/region/industry relevant to the question.  
# Action: print("Relevant rows preview:", df[df.index.astype(str).str.contains(','.join({cell_keywords}), na=False)].to_string())  

# Step 3: Extract target data (avoid calculation unless required)  
# Thought: Use confirmed columns (Step1) and rows (Step2) to extract the value directly (no unnecessary division; "proportion" often = "percent" in table).  
# Action: print("Target value:", df.loc[REPLACE_WITH_ROW_LABEL, REPLACE_WITH_COLUMN_TUPLE])  

# Step 4: Verify consistency with question  
# Thought: Check if the extracted value matches the question's requirement (e.g., "all countries" → calculate mean; "specific region" → single value).  
# Action: print("Verification: Value matches question scope?", True/False)  

# ### Strict Response Format  
# Thought: [Current step from the flow, e.g., "Step1: Parse schema to find target columns"]  
# Action: [Single-line Python command for the step]  
# Observation: [Result of the Action]  
# ... (Repeat Thought/Action/Observation for Steps 1-4)  
# Thought: [Verify: Link observations to question—confirm columns/rows are correct, value matches scope, no calculation errors]  
# Final Answer: [numerical value or string matching the question]  

# ### Universal Rules (Apply to All Tables)  
# 1. Multi-level columns: Always use tuple indexing (e.g., ("northern ontario", "other workers")), never custom column names.  
# 2. No Calculation:  The Final Answer must not require any calculations (including division, multiplication, summation, averaging, etc.) .
# 3. Cell keywords: Use {cell_retrieval_result} to find rows (e.g., "food service" for restaurant sector).  
# 4. No formatting: Avoid markdown, quotes, or extra text in Final Answer.  

# Now, use {schema_retrieval_result} and {cell_retrieval_result} to answer the question: {query}  

# Begin!  
# '''

# tablerag_solve_table_prompt = '''  
# You are working with a pandas dataframe regarding "{table_caption}" in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question: {query}  

# Tool description:  
# - `python_repl_ast`: A Python interactive shell. Use this to execute python commands. Input must be a valid single-line python command (no natural language).  

# Critical Context:  
# You cannot view the full table, but {schema_retrieval_result} and {cell_retrieval_result} provide column schemas (e.g., multi-level headers) and key cell values. Use these to locate data.  

# ### Universal Reasoning Flow (MUST FOLLOW IN ORDER)  
# Step 1: Parse column schema to confirm data location  
# Thought: Analyze {schema_retrieval_result} to identify target columns (use tuple for multi-level headers, e.g., ("gender", "education")).  
# Action: print("Target columns to confirm:", df.columns.tolist())  

# Step 2: Locate rows with key cell values  
# Thought: Match {cell_retrieval_result} with dataframe rows to find the category/region/industry relevant to the question. Key keyword for row: "food service" (equals "restaurant and food services sector").  
# Action: print("Relevant rows preview:", df[df.index.astype(str).str.contains(','.join({cell_retrieval_result}), na=False)].to_string())  

# Step 3: Extract target data (ABSOLUTELY NO CALCULATIONS)  
# Thought: Use confirmed columns (Step1) and rows (Step2) to extract the value DIRECTLY. In the table, "proportion" in the question IS EXACTLY the "percent" value shown—NO division (e.g., /100), multiplication, or any arithmetic allowed.  
# Action: print("Target value:", df.loc[REPLACE_WITH_ROW_LABEL, REPLACE_WITH_COLUMN_TUPLE])  

# Step 4: Verify consistency with question  
# Thought: Check if the extracted value is the original table value (no calculations) and matches the scope (e.g., "northern ontario" + "food service" + "other workers").  
# Action: print("Verification: Original table value without calculations?", True/False)  

# ### Strict Response Format  
# Thought: [Current step from the flow, e.g., "Step1: Parse schema to find target columns"]  
# Action: [Single-line Python command for the step—NO arithmetic operators (+, -, *, /)]  
# Observation: [Result of the Action]  
# ... (Repeat Thought/Action/Observation for Steps 1-4)  
# Thought: [Verify: Confirm 3 points—1. Column tuple (e.g., ("northern ontario", "other workers")) is correct; 2. Row label (e.g., "food service") is correct; 3. Extracted value is original (no /100 etc.), which matches "proportion" requirement.]  
# Final Answer: [Original numerical value from the table—DO NOT change decimals (e.g., 58.1 instead of 0.581)]  

# ### Universal Rules (MANDATORY—VIOLATION = WRONG ANSWER)  
# 1. Multi-level columns: Always use tuple indexing (e.g., ("northern ontario", "other workers")), never custom column names or single-level indexes.  
# 2. Absolute No Calculation: The Final Answer must be the EXACT original value from the table. NO division (especially /100), multiplication, summation, averaging, or any arithmetic operations—even if you think "proportion needs it".  
# 3. Proportion = Percent: In all tables, "proportion" in the question directly equals the "percent" value in the table (e.g., table value 58.1 = Final Answer 58.1, not 0.581).  
# 4. Cell keywords: Use {cell_retrieval_result} (e.g., "food service", "other workers") to locate rows/columns—never guess labels.  
# 5. No formatting: Avoid markdown, quotes, extra text, or value modifications in Final Answer.  

# Now, use {schema_retrieval_result} and {cell_retrieval_result} to answer the question: {query}  

# Begin!  
# '''

# tablerag_solve_table_prompt = '''  
# You are working with a pandas dataframe named `df` about "{table_caption}". Answer {query} using `python_repl_ast` (only 1-line valid Python commands; no natural language).  

# ### Critical Preparations (MUST DO FIRST)  
# The table has unknown rows/columns. You MUST first diagnose its structure via commands—never guess labels/indexes.  

# ### Mandatory 4-Step Universal Workflow (All Tables Adaptable)  
# Step 1: Diagnose multi-level columns (core to avoid KeyError)  
# Thought: Step1: Must check column hierarchy (1st/2nd level labels) to confirm [region/category] (1st) and [group/type] (2nd)  
# Action: print("Multi-level columns (1st→2nd):", df.columns.tolist())  

# Step 2: Map question keywords to table labels  
# Thought: Step2: Link question terms to table content (e.g., "restaurant sector" → "food service"; "proportion" → "percent")  
# Action: print("Question keywords mapped:", {cell_retrieval_result})  # 用检索到的单元格关键词匹配  

# Step 3: Locate target row (by mapped content in Step2)  
# Thought: Step3: Find row index/label for the mapped category (e.g., "food service") via the first column  
# Action: print("Target row info:", df[df.iloc[:,0].astype(str).str.contains("MAPPED_CATEGORY", na=False)].to_string())  # 替换MAPPED_CATEGORY为Step2结果  

# Step 4: Extract original value (NO calculations allowed)  
# Thought: Step4: Use "row index/label + column tuple (from Step1)" to get value; "proportion" = table's percent value (no /100)  
# Action: print("Target value:", df.iloc[ROW_INDEX, df.columns.get_loc(("COL1_LABEL", "COL2_LABEL"))] if df.columns.nlevels>1 else df.iloc[ROW_INDEX, df.columns.get_loc("COL_LABEL")])  

# ### Strict Error Correction Logic (If Any Observation Has Error)  
# 1. **KeyError (column)**:  
#    - If "('A','B')" fails → Recheck Step1: Column tuple must be (1st level label, 2nd level label), not reversed.  
#    - If single label fails → Check if columns are multi-level (Step1 shows tuples) and use tuple index.  
# 2. **KeyError (row)**:  
#    - Replace "contains" with "== " in Step3 if exact match is needed; ensure mapped term is correct (Step2).  
# 3. **TypeError (calculation)**:  
#    - Delete all +-*/ operations immediately—Final Answer is Step4's original value.  

# ### Response Format (Strictly Follow)  
# Thought: [Step's thought, e.g., "Step1: Diagnose column hierarchy to avoid KeyError"]  
# Action: [Step's command, replace placeholders after prior steps]  
# Observation: [Action result]  
# ... (Repeat Steps 1-4; add error correction if needed)  
# Thought: Verify: 1. Column/row labels match Step1/3 results; 2. Value is original (no calculations); 3. Matches question's "proportion/percent" requirement.  
# Final Answer: [Original value from Step4 (e.g., 55.0, 78.1)—no changes]  

# ### Universal Iron Rules (All Tables Apply)  
# 1. Column Index: Multi-level columns → MUST use tuple (1st level, 2nd level); single-level → use single label.  
# 2. Absolute No Calculation: Proportion = Percent: In all tables, "proportion" in the question directly equals the "percent" value in the table (e.g., table value 58.1 = Final Answer 58.1, not 0.581).
# 3. Row Location: Always find rows via the first column (df.iloc[:,0])—never mix rows with columns.  
# 4. No Guessing: If errors persist, re-run Step1 (recheck structure)—never output random values.  

# Table Context:  
# Schema: {schema_retrieval_result}  
# Key Cells: {cell_retrieval_result}  

# Answer {query}  
# Begin!  
# '''

# tablerag_solve_table_prompt = '''  
# You are working with a pandas dataframe named `df` about "{table_caption}". Answer {query} using `python_repl_ast` (only 1-line valid Python commands; no natural language).  

# ### ⚠️  FATAL ERROR WARNING (VIOLATION = WRONG ANSWER)  
# 1. NO CALCULATIONS: "proportion" = table's original percent value. NEVER use /100, /, *, +, -—Final Answer is the table's number directly.  
# 2. COLUMN RULE: Multi-level columns → use tuple (level1, level2) (e.g., (region, worker_type)); single-level → use single label. NEVER use row content as column name.  
# 3. ROW RULE: Find rows via first column (df.iloc[:,0])—e.g., "food service" is in the first column, not columns.  

# ### Mandatory 4-Step Universal Workflow (All Tables Adaptable)  
# Step 1: Diagnose column structure (fix KeyError)  
# Thought: Step1: Must check column labels (multi-level/single-level) to avoid wrong indices  
# Action: print("Column structure:", df.columns.tolist())  

# Step 2: Locate target row (by first column content)  
# Thought: Step2: Map question's "restaurant and food services sector" → "food service" (from {cell_retrieval_result}), find its row index  
# Action: print("Target row index:", df[df.iloc[:,0].astype(str).str.contains("food service", na=False)].index[0])  

# Step 3: Confirm target column (by question scope)  
# Thought: Step3: Column = (region, group) from question. Region: "northern ontario"; Group: "other agri-food workers" → "other workers" (from {cell_retrieval_result})  
# Action: print("Target column tuple:", ("northern ontario", "other workers"))  

# Step 4: Extract original value + verify (no calculation)  
# Thought: Step4: Use row index (Step2) + column tuple (Step3) to get value; MUST NOT calculate  
# Action: print("Target value:", df.iloc[STEP2_INDEX, df.columns.get_loc(STEP3_TUPLE)] if df.columns.nlevels>1 else df.iloc[STEP2_INDEX, df.columns.get_loc(STEP3_LABEL)])  

# ### 🔍  Mandatory Error Correction (If Any Observation Has Error)  
# - If KeyError (column): Re-run Step1 → column tuple must be (level1, level2), not row content/guessed names.  
# - If KeyError (row): Replace "contains" with "==" in Step2 (exact match), check {cell_retrieval_result} for correct row content.  
# - If TypeError (calculation): Delete all +-*/ immediately, re-extract original value.  

# ### Strict Response Format  
# Thought: [Step's thought + error warning (if needed), e.g., "Step1: Check columns to avoid KeyError; remember no /100"]  
# Action: [Step's command, replace STEP2_INDEX/STEP3_TUPLE after prior steps]  
# Observation: [Action result]  
# ... (Repeat Steps 1-4; run error correction if needed)  
# Thought: Verify: 1. Value is from Step4 (no calculations); 2. Column tuple/row index match Step1-3; 3. Matches question's "proportion" requirement.  
# Final Answer: [Step4's original value (e.g., 58.1), no changes]  

# ### Universal Rules (All Tables)  
# 1. Schema Anchor: Use {schema_retrieval_result} to confirm column levels, {cell_retrieval_result} to map question terms to table content.  
# 2. No Guessing: If errors persist, re-run Step1 (column diagnosis)—never invent labels/indexes.  
# 3. Result Check: Ensure Final Answer has the same decimal places as Step4's value (e.g., 58.1≠0.581).  

# Table Context:  
# Schema: {schema_retrieval_result}  
# Key Cells: {cell_retrieval_result}  

# Answer {query}  
# Begin!  
# '''

tablerag_solve_table_prompt = '''  
You are working with a pandas dataframe named `df` about "{table_caption}". Answer {query} using `python_repl_ast` (only 1-line valid Python commands; no natural language).  

### ⚠️  FATAL LOGIC ERROR WARNING  
The question asks for "proportion of [Group A] in [Sector B]" → THIS IS DIRECTLY THE PERCENT VALUE IN THE TABLE FOR "[Sector B] row + [Group A] column".  
- Example: "What proportion of workers in food service (Sector B) are other workers (Group A) in northern ontario?" → Find the cell where row="food service" and column=("northern ontario", "other workers") → that value is the answer.  
- NEVER calculate A/B, B/A, or divide by 100. The table's number is the proportion.  

### Mandatory 4-Step Workflow (Lock Logic)  
Step 1: Identify [Sector B] in the question (e.g., "restaurant and food services sector")  
Thought: Step1: [Sector B] = "restaurant and food services sector" → map to table row: "food service" (from {cell_retrieval_result})  
Action: print("Sector B row content:", "food service")  

Step 2: Find row index of [Sector B] (must use first column)  
Thought: Step2: Locate row index where first column (df.iloc[:,0]) = "food service"  
Action: print("Sector B row index:", df[df.iloc[:,0] == "food service"].index[0])  

Step 3: Identify [Group A] and region in the question (e.g., "other agri-food workers" in "northern ontario")  
Thought: Step3: [Group A] = "other agri-food workers" → map to table column: "other workers"; Region = "northern ontario" → column tuple: ("northern ontario", "other workers")  
Action: print("Target column tuple:", ("northern ontario", "other workers"))  

Step 4: Extract the value (ABSOLUTELY NO calculations)  
Thought: Step4: Use row index (Step2) + column tuple (Step3) → this is the proportion (no division/multiplication)  
Action: print("Proportion value:", df.iloc[STEP2_INDEX, df.columns.get_loc(STEP3_TUPLE)])  

### Error Correction (Must Trigger)  
- If KeyError (column): Recheck column tuple format → ("region", "group") (e.g., ("northern ontario", "other workers")), never reverse.  
- If KeyError (row): Ensure Step2 uses "df.iloc[:,0] == 'food service'" (exact match), not column names.  
- If calculation detected (+-*/): Delete all operations immediately, re-extract with Step4.  

### Response Format  
Thought: [Step's thought + logic reminder, e.g., "Step1: [Sector B] is food service (no calculation)"]  
Action: [Step's command, replace STEP2_INDEX after Step2]  
Observation: [Result]  
... (Steps 1-4)  
Thought: Verify: 1. Value is from "food service" row + ("northern ontario", "other workers") column; 2. No calculations; 3. Matches "proportion of Group A in Sector B".  
Final Answer: [Step4's value (e.g., 58.1)]  

### Universal Rules  
1. Logic Lock: "proportion of A in B" = table value at "B row + A column" → no math.  
2. Column Rule: Multi-level columns require tuple ("region", "group") → no other format.  
3. Row Rule: Rows are found via first column (df.iloc[:,0]) → never use regions as row labels.  

Table Context:  
Schema: {schema_retrieval_result}  
Key Cells: {cell_retrieval_result}  

Answer {query}  
Begin!  
'''