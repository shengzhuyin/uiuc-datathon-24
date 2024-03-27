## UIUC Stats + Synchrony Datathon 2024: Team Init To Win It

This was Team Init To Win It's submission for the Datathon. 

Member: Shengzhu Yin, Tanmoy Debnath, Anusha Verma Chandraju, Urmi Gori

File descriptions:
 - **pre_processing.ipynb**: For pre-processing the provided datasets, and then performing EDA and various other analyses
 - **combined_feature_and_sequence_importance.ipynb**: For identifying features, as individuals and in sequence, which were most impactful, using Logistic Regression, Support Vector Machine, Ridge Classifier, and XGBoost.
 - **Synchrony_data.ipynb**: Main chunck that includes most analysis except for machine learning (in **xgboost.py**) and deep learning (in **combined_feature_and_sequence_importance.ipynb**)
 - **visualization_.ipynb**: A Radial graph in contention for best visualization. Used for Account / e-Bill / Card Activation status Vs. reason labels in a loop radial graphing format.
 - **modeling.ipynb**: For modeling and subsequent predictions, using LSTM and Transformers. We also provided an LLM-based solution, testing Google's BERT in the process, but settling with OpenAI's GPT-3.5 Turbo API.
 - **xgboostcd.py**: Used XGBoost to extract the relative importance of attributes used to infer resolved/floor calls. Requires pre-processing of categorical data (reason, most) to binary encoding.
 - **Presentation.pdf**: Presentation Slides
