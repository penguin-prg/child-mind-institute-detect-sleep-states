class Features:
    """
    作成した特徴量のリストを管理するクラス
    NFL2022で雑に作ったやつ。
    TODO: もっとちゃんとしたのを作る。
    """

    def __init__(self):
        self._num_features = set()
        self._cat_features = set()

    def add_num_feature(self, feature: str):
        """
        数値特徴量を1つ追加
        """
        assert type(feature) is str
        self._num_features.add(feature)

    def add_num_features(self, features: list):
        """
        数値特徴量を複数を追加
        """
        assert type(features) is list
        self._num_features |= set(features)

    def add_cat_feature(self, feature: str):
        """
        カテゴリカルな特徴量を1つ追加
        """
        assert type(feature) is str
        self._cat_features.add(feature)

    def add_cat_features(self, features: list):
        """
        カテゴリカルな特徴量を1つ複数
        """
        assert type(features) is list
        self._cat_features |= set(features)

    def num_features(self):
        """
        数値特徴量をソートして取得
        """
        return sorted(list(self._num_features))

    def cat_features(self):
        """
        カテゴリカル特徴量をソートして取得
        """
        return sorted(list(self._cat_features))

    def all_features(self):
        """
        全ての特徴量を取得
        """
        return self.num_features() + self.cat_features()

    def clear(self):
        """
        特徴量を削除
        """
        self._num_features = set()
        self._cat_features = set()

    def remove_num_features(self, features: list):
        """
        指定した数値特徴量を削除
        """
        assert type(features) is list
        self._num_features -= set(features)

    def remove_cat_features(self, features: list):
        """
        指定したカテゴリカル特徴量を削除
        """
        assert type(features) is list
        self._cat_features -= set(features)

    def __str__(self) -> str:
        return f"[num_features]\n{self.num_features()}\n[cat_features]\n{self.cat_features()}"
