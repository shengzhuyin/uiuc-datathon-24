## UIUC Stats + Synchrony Datathon 2024: Team Init To Win It

This was Team Init To Win It's submission for the Datathon. 

Member: Shengzhu Yin, Tanmoy Debnath, Anusha Verma Chandraju, Urmi Gori

File descriptions:
 - **combined_feature_and_sequence_importance.ipynb**: For identifying features, as individuals and in sequence, which were most impactful, using Logistic Regression, Support Vector Machine, Ridge Classifier, and XGBoost.
 - **Synchrony_data.ipynb**: Main file that includes most pre-processing, EDA, and other analysis except for machine learning (in **xgboost.py**) and deep learning (in **combined_feature_and_sequence_importance.ipynb**)
 - **visualization_.ipynb**: A Radial graph in contention for best visualization. Used for Account / e-Bill enrollment / Card activation status Vs. reason labels in a loop radial graphing format.
 - **modeling.ipynb**: For modeling and subsequent predictions, using LSTM and Transformers. We also provided an LLM-based solution, testing Google's BERT in the process, but settling with OpenAI's GPT-3.5 Turbo API.
 - **xgboostcd.py**: Used XGBoost to extract the relative importance of attributes used to infer resolved/floor calls. Requires pre-processing of categorical data (reason, most) to binary encoding.
 - **Presentation.pdf**: Presentation Slides
 - **script**: many fragments of codes during the development, most of the codes here are condensed to **combined_feature_and_sequence_importance.ipynb**
 - **dictionary**: dictionary for account status / e-Bill enrollment status / Card activation status and etc.
