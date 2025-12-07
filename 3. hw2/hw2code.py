import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    if len(feature_vector) == 0:
        return np.array([]), np.array([]), None, None
    
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]
    
    unique_features = np.unique(sorted_features)
    if len(unique_features) <= 1:
        return np.array([]), np.array([]), None, None
    
    thresholds = (unique_features[:-1] + unique_features[1:]) / 2.0
    
    n_total = len(feature_vector)
    n_left = np.arange(1, len(unique_features))
    
    left_mask = sorted_features[:, None] < thresholds[None, :]
    left_counts_1 = np.sum((sorted_targets[:, None] == 1) & left_mask, axis=0)
    left_counts_0 = np.sum((sorted_targets[:, None] == 0) & left_mask, axis=0)
    right_counts_1 = np.sum(sorted_targets == 1) - left_counts_1
    right_counts_0 = np.sum(sorted_targets == 0) - left_counts_0
    
    left_sizes = left_counts_1 + left_counts_0
    right_sizes = right_counts_1 + right_counts_0
    
    valid_mask = (left_sizes > 0) & (right_sizes > 0)
    
    if not np.any(valid_mask):
        return np.array([]), np.array([]), None, None
    
    thresholds = thresholds[valid_mask]
    left_counts_1 = left_counts_1[valid_mask]
    left_counts_0 = left_counts_0[valid_mask]
    right_counts_1 = right_counts_1[valid_mask]
    right_counts_0 = right_counts_0[valid_mask]
    left_sizes = left_sizes[valid_mask]
    right_sizes = right_sizes[valid_mask]
    
    p1_left = left_counts_1 / left_sizes
    p0_left = left_counts_0 / left_sizes
    p1_right = right_counts_1 / right_sizes
    p0_right = right_counts_0 / right_sizes
    
    H_left = 1 - p1_left**2 - p0_left**2
    H_right = 1 - p1_right**2 - p0_right**2
    
    ginis = -(left_sizes / n_total) * H_left - (right_sizes / n_total) * H_right
    
    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
    
    def get_params(self, deep=True):
        """Возвращает параметры для совместимости с sklearn"""
        return {
            'feature_types': self._feature_types.copy() if isinstance(self._feature_types, list) else self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }
    
    def set_params(self, **params):
        """Устанавливает параметры для совместимости с sklearn"""
        if 'feature_types' in params:
            self._feature_types = params['feature_types']
        if 'max_depth' in params:
            self._max_depth = params['max_depth']
        if 'min_samples_split' in params:
            self._min_samples_split = params['min_samples_split']
        if 'min_samples_leaf' in params:
            self._min_samples_leaf = params['min_samples_leaf']
        return self
    
    def __sklearn_clone__(self):
        """Кастомное клонирование для совместимости с sklearn"""
        # Создаем новый экземпляр с теми же параметрами
        feature_types = self._feature_types.copy() if isinstance(self._feature_types, list) else self._feature_types
        cloned = DecisionTree(
            feature_types=feature_types,
            max_depth=self._max_depth,
            min_samples_split=self._min_samples_split,
            min_samples_leaf=self._min_samples_leaf
        )
        return cloned

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    if current_count > 0:
                        ratio[key] = current_click / current_count
                    else:
                        ratio[key] = 0
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None:
                continue
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [key for key, val in categories_map.items() if val < threshold]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        left_mask = split
        right_mask = np.logical_not(split)
        
        if self._min_samples_leaf is not None:
            if np.sum(left_mask) < self._min_samples_leaf or np.sum(right_mask) < self._min_samples_leaf:
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[left_mask], sub_y[left_mask], node["left_child"], depth + 1)
        self._fit_node(sub_X[right_mask], sub_y[right_mask], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        if node["type"] == "terminal":
            return node["class"]
        
        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]
        
        if feature_type == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
