import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import time
import great_expectations as gx
import json
from pathlib import Path
import sys


class DataLoader:
    """データロードを行うクラス"""

    @staticmethod
    def load_titanic_data(path=None):
        """Titanicデータセットを読み込む"""
        if path:
            return pd.read_csv(path)
        else:
            # ローカルのファイル
            local_path = r"C:\Users\Yokono Haruhiko\cursor\AIE\lecture-ai-engineering\day5\演習2\data\Titanic.csv"
            if os.path.exists(local_path):
                return pd.read_csv(local_path)

    @staticmethod
    def preprocess_titanic_data(data):
        """Titanicデータを前処理する"""
        # 必要な特徴量を選択
        data = data.copy()

        # 不要な列を削除
        columns_to_drop = []
        for col in ["PassengerId", "Name", "Ticket", "Cabin"]:
            if col in data.columns:
                columns_to_drop.append(col)

        if columns_to_drop:
            data.drop(columns_to_drop, axis=1, inplace=True)

        # 目的変数とその他を分離
        if "Survived" in data.columns:
            y = data["Survived"]
            X = data.drop("Survived", axis=1)
            return X, y
        else:
            return data, None


class DataValidator:
    """データバリデーションを行うクラス"""

    @staticmethod
    def validate_titanic_data(data):
        """Titanicデータセットの検証"""
        # DataFrameに変換
        if not isinstance(data, pd.DataFrame):
            return False, ["データはpd.DataFrameである必要があります"]

        # Great Expectationsを使用したバリデーション
        try:
            context = gx.get_context()
            # オリジナル（v0.18.22以前）のコード（コメントアウト）
            # data_source = context.data_sources.add_pandas("pandas")
            # data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")
            # batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
            # batch = batch_definition.get_batch(batch_parameters={"dataframe": data})

            # v0.18.22用の新しいコード
            context.add_datasource(
                name="my_pandas_datasource",
                class_name="Datasource",
                execution_engine={"class_name": "PandasExecutionEngine"},
                data_connectors={
                    "default_runtime_data_connector_name": {
                        "class_name": "RuntimeDataConnector",
                        "batch_identifiers": ["default_identifier_name"],
                    }
                },
            )

            # batch_request = {
            #    "datasource_name": "my_pandas_datasource",
            #    "data_connector_name": "default_runtime_data_connector_name",
            #    "data_asset_name": "my_data_asset",
            #    "runtime_parameters": {"batch_data": data},
            #    "batch_identifiers": {"default_identifier_name": "default_id"},
            # }
            # batch = context.get_batch(batch_request)
            # v0.18.22用の新しいコード
            batch = context.get_validator(
                datasource_name="my_pandas_datasource",
                data_connector_name="default_runtime_data_connector_name",
                data_asset_name="my_data_asset",
                runtime_parameters={"batch_data": data},
                batch_identifiers={"default_identifier_name": "default_id"},
            )

            results = []
            # debug
            print(batch)

            # 必須カラムの存在確認
            required_columns = [
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Fare",
                "Embarked",
            ]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]
            if missing_columns:
                print(f"警告: 以下のカラムがありません: {missing_columns}")
                return False, [{"success": False, "missing_columns": missing_columns}]

            # 旧いコード（コメントアウト）
            # expectations = [
            #     gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            #         column="Pclass", value_set=[1, 2, 3]
            #     ),
            #     gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            #         column="Sex", value_set=["male", "female"]
            #     ),
            #     gx.expectations.ExpectColumnValuesToBeBetween(
            #         column="Age", min_value=0, max_value=100
            #     ),
            #     gx.expectations.ExpectColumnValuesToBeBetween(
            #         column="Fare", min_value=0, max_value=600
            #     ),
            #     gx.expectations.ExpectColumnDistinctValuesToBeInSet(
            #         column="Embarked", value_set=["C", "Q", "S", ""]
            #     ),
            # ]

            # v0.18.22用の新しいコード
            results.append(
                batch.expect_column_distinct_values_to_be_in_set(
                    column="Pclass", value_set=[1, 2, 3]
                )
            )
            results.append(
                batch.expect_column_distinct_values_to_be_in_set(
                    column="Sex", value_set=["male", "female"]
                )
            )
            results.append(
                batch.expect_column_values_to_be_between(
                    column="Age", min_value=0, max_value=100
                )
            )
            results.append(
                batch.expect_column_values_to_be_between(
                    column="Fare", min_value=0, max_value=600
                )
            )
            results.append(
                batch.expect_column_distinct_values_to_be_in_set(
                    column="Embarked", value_set=["C", "Q", "S", ""]
                )
            )

            # すべての検証が成功したかチェック
            is_successful = all(result.success for result in results)
            return is_successful, results

        except Exception as e:
            print(f"Great Expectations検証エラー: {e}")
            return False, [{"success": False, "error": str(e)}]


class ModelTester:
    """モデルテストを行うクラス"""

    @staticmethod
    def create_preprocessing_pipeline():
        """前処理パイプラインを作成"""
        numeric_features = ["Age", "Fare", "SibSp", "Parch"]
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_features = ["Pclass", "Sex", "Embarked"]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",  # 指定されていない列は削除
        )
        return preprocessor

    @staticmethod
    def train_model(X_train, y_train, model_params=None):
        """モデルを学習する"""
        if model_params is None:
            model_params = {"n_estimators": 100, "random_state": 42}

        # 前処理パイプラインを作成
        preprocessor = ModelTester.create_preprocessing_pipeline()

        # モデル作成
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(**model_params)),
            ]
        )

        # 学習
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """モデルを評価する"""
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        return {"accuracy": accuracy, "inference_time": inference_time}

    @staticmethod
    def save_model(model, path="models/titanic_model.pkl"):
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"titanic_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        return path

    @staticmethod
    def load_model(path="models/titanic_model.pkl"):
        """モデルを読み込む"""
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def compare_with_baseline(current_metrics, baseline_threshold=0.75):
        """ベースラインと比較する"""
        return current_metrics["accuracy"] >= baseline_threshold

    @staticmethod
    def save_metrics(metrics, output_dir="validation_results"):
        """メトリクスをJSONファイルとして保存"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # メトリクスを保存
        metrics_data = {
            "accuracy": metrics["accuracy"],
            "inference_time": metrics["inference_time"],
        }

        output_file = output_path / "latest_metrics.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)

        return str(output_file)

    @staticmethod
    def compare_with_previous(metrics, previous_metrics_path):
        """現在のメトリクスと過去のメトリクスを比較"""
        if not Path(previous_metrics_path).exists():
            return {"has_previous": False, "message": "No previous metrics found"}

        with open(previous_metrics_path, "r", encoding="utf-8") as f:
            previous_metrics = json.load(f)

        accuracy_change = metrics["accuracy"] - previous_metrics["accuracy"]
        time_change = metrics["inference_time"] - previous_metrics["inference_time"]

        return {
            "has_previous": True,
            "current_accuracy": metrics["accuracy"],
            "previous_accuracy": previous_metrics["accuracy"],
            "accuracy_change": accuracy_change,
            "current_time": metrics["inference_time"],
            "previous_time": previous_metrics["inference_time"],
            "time_change": time_change,
            "is_improved": accuracy_change > 0 and time_change <= 0,
        }


# テスト関数（pytestで実行可能）
def test_data_validation():
    """データバリデーションのテスト"""
    # データロード
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)

    # 正常なデータのチェック
    success, results = DataValidator.validate_titanic_data(X)
    assert success, "データバリデーションに失敗しました"

    # 異常データのチェック
    bad_data = X.copy()
    bad_data.loc[0, "Pclass"] = 5  # 明らかに範囲外の値
    success, results = DataValidator.validate_titanic_data(bad_data)
    assert not success, "異常データをチェックできませんでした"


def test_model_performance():
    """モデル性能のテスト"""
    # データ準備
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデル学習
    model = ModelTester.train_model(X_train, y_train)

    # 評価
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    # ベースラインとの比較
    assert ModelTester.compare_with_baseline(
        metrics, 0.75
    ), f"モデル性能がベースラインを下回っています: {metrics['accuracy']}"

    # 推論時間の確認
    assert (
        metrics["inference_time"] < 1.0
    ), f"推論時間が長すぎます: {metrics['inference_time']}秒"


def run_workflow_validation():
    """ワークフロー用の検証実行"""
    # データの準備
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルの学習と評価
    model = ModelTester.train_model(X_train, y_train)
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    # メトリクスの保存
    metrics_file = ModelTester.save_metrics(metrics)

    # 過去の結果との比較
    previous_metrics_path = Path("validation_results/latest_metrics.json")
    comparison = ModelTester.compare_with_previous(metrics, previous_metrics_path)

    # GitHub Actions用の出力
    if comparison["has_previous"]:
        print(f"::set-output name=has_previous::true")
        print(
            f"::set-output name=current_accuracy::{comparison['current_accuracy']:.4f}"
        )
        print(
            f"::set-output name=previous_accuracy::{comparison['previous_accuracy']:.4f}"
        )
        print(f"::set-output name=accuracy_change::{comparison['accuracy_change']:.4f}")
        print(f"::set-output name=current_time::{comparison['current_time']:.4f}")
        print(f"::set-output name=previous_time::{comparison['previous_time']:.4f}")
        print(f"::set-output name=time_change::{comparison['time_change']:.4f}")
        print(
            f"::set-output name=is_improved::{str(comparison['is_improved']).lower()}"
        )

        # 比較結果の表示
        print("\nモデル性能の比較:")
        print(
            f"精度: {comparison['current_accuracy']:.4f} (前回: {comparison['previous_accuracy']:.4f}, 変化: {comparison['accuracy_change']:+.4f})"
        )
        print(
            f"推論時間: {comparison['current_time']:.4f}秒 (前回: {comparison['previous_time']:.4f}秒, 変化: {comparison['time_change']:+.4f}秒)"
        )
        print(f"改善: {'あり' if comparison['is_improved'] else 'なし'}")
    else:
        print("::set-output name=has_previous::false")
        print("\n初回実行のため、比較は行いません")

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--workflow":
        sys.exit(run_workflow_validation())

    # データロード
    data = DataLoader.load_titanic_data()
    X, y = DataLoader.preprocess_titanic_data(data)

    # データバリデーション
    success, results = DataValidator.validate_titanic_data(X)
    print(f"データ検証結果: {'成功' if success else '失敗'}")

    for result in results:
        # "success": falseの場合はエラーメッセージを表示
        if not result["success"]:
            print(f"異常タイプ: {result['expectation_config']['type']}, 結果: {result}")
    if not success:
        print("データ検証に失敗しました。処理を終了します。")
        exit(1)

    # モデルのトレーニングと評価
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # パラメータ設定
    model_params = {"n_estimators": 100, "random_state": 42}

    # モデルトレーニング
    model = ModelTester.train_model(X_train, y_train, model_params)
    metrics = ModelTester.evaluate_model(model, X_test, y_test)

    print(f"精度: {metrics['accuracy']:.4f}")
    print(f"推論時間: {metrics['inference_time']:.4f}秒")

    # モデル保存
    model_path = ModelTester.save_model(model)

    # ベースラインとの比較
    baseline_ok = ModelTester.compare_with_baseline(metrics)
    print(f"ベースライン比較: {'合格' if baseline_ok else '不合格'}")
