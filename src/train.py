import argparse
import mlflow
import optuna
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from src.data_prep import load_raw_data, clean_mostly_null_cols, clean_mostly_null_rows, find_merge_cols
from src.transformers import ChiSquareSelector, VIFSelector, AnovaSelector, EduEncoder, build_onehot_step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vif_threshold', type=float, default=6)
    parser.add_argument('--chi2_alpha', type=float, default=0.05)
    parser.add_argument('--anova_alpha', type=float, default=0.05)
    parser.add_argument('--drop_credit_score', action='store_true')
    #parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--test_size', type=float, default=0.2)
    return parser.parse_args()


def load_and_clean():
    df1, df2 = load_raw_data(
        r"C:\Users\Lenovo\Downloads\credit_card\case_study1.xlsx",
        r"C:\Users\Lenovo\Downloads\credit_card\case_study2.xlsx"
    )
    df1 = df1[df1['Age_Oldest_TL'] != -99999]
    df2 = clean_mostly_null_cols(df2)
    df2 = clean_mostly_null_rows(df2)
    merge_cols = find_merge_cols(df1, df2)
    return df1.merge(df2, on=merge_cols, how='inner')


def build_pipeline(args, cat_cols,trial):
    params = {
        "model__n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "model__max_depth": trial.suggest_int("max_depth", 3, 10),
    }

    pipe=Pipeline(steps=[
        ('chi2_filter', ChiSquareSelector(alpha=args.chi2_alpha)),
        ('vif_filter', VIFSelector(threshold=args.vif_threshold)),
        ('anova_filter', AnovaSelector(alpha=args.anova_alpha)),
        ('education_encode', EduEncoder(col='EDUCATION')),
        ('onehot', build_onehot_step(cat_cols)),
        ('model', RandomForestClassifier(random_state=42)),
    ])
    pipe.set_params(**params)
    return pipe
def objective(trial,X_train,X_test,y_train,y_test,cat_cols,args):
    with mlflow.start_run(nested=True):
        pipe=build_pipeline(args,cat_cols=cat_cols,trial=trial)
        pipe.fit(X_train,y_train)
        y_pred=pipe.predict(X_test)
        acc=accuracy_score(y_test,y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        mlflow.log_params(trial.params)
        mlflow.log_metrics({'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1})

        trial.set_user_attr('precision',precision)
        trial.set_user_attr('recall',recall)
        trial.set_user_attr('f1',f1)
    return acc

    
def main():
    args = parse_args()
    df = load_and_clean()

    target = 'Approved_Flag'
    y = df[target]
    X = df.drop(columns=[target, 'PROSPECTID'])
    if args.drop_credit_score:
        X = X.drop(columns=['Credit_Score'])
    le =LabelEncoder()
    y_encoded=le.fit_transform(y)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=args.test_size, random_state=42
    )
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    cat_cols = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
    

    mlflow.set_experiment('Credit card analysis')

    # run_name='without_credit_score' if args.drop_credit_score  else 'with_credit_score'
    # with mlflow.start_run(run_name=run_name,nested=True):
    #     #mlflow.log_params(vars(args))

    #     pipe.fit(X_train, y_train)
    #     y_pred = pipe.predict(X_test)

    #     acc = accuracy_score(y_test, y_pred)

    #     precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
    #    # mlflow.log_metrics({'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1})

    #     mlflow.sklearn.log_model(
    #         pipe,
    #         artifact_path='model',
    #         skops_trusted_types=[
    #             'sklearn.tree._tree.Tree',
    #             'src.transformers.AnovaSelector',
    #             'src.transformers.ChiSquareSelector',
    #             'src.transformers.EduEncoder',
    #             'src.transformers.VIFSelector',
    #         ],
    #     )
    with mlflow.start_run(run_name="optuna_study_on_credit_card"):   # parent run
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, X_train, X_test, y_train, y_test, cat_cols, args),
            n_trials=20
        )

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_accuracy", study.best_value)
        mlflow.log_metric("best_precision", study.best_trial.user_attrs["precision"])
        mlflow.log_metric("best_recall", study.best_trial.user_attrs["recall"])
        mlflow.log_metric("best_f1", study.best_trial.user_attrs["f1"])

        best_pipe = build_pipeline(args, cat_cols, optuna.trial.FixedTrial(study.best_params))
        best_pipe.fit(X_train, y_train)
        mlflow.sklearn.log_model(best_pipe, artifact_path="model",skops_trusted_types=[
                'sklearn.tree._tree.Tree',
                'src.transformers.AnovaSelector',
                'src.transformers.ChiSquareSelector',
                'src.transformers.EduEncoder',
                'src.transformers.VIFSelector',
            ],)


if __name__ == "__main__":
    main()
