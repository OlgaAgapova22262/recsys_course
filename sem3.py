"""
Семинар 3. Контентная фильтрация
Цель: Разработать методы контентной фильтрации по пользователям и по фильмам.
В качестве контента используем описание жанров для каждого фильма из movies.csv.
Для векторизации жанров используем CountVectorizer с разделителем "|".
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import build_user_item_matrix, id_to_movie, load_data, print_user_rated_items


class ContentRecommender:
    """
    Класс для построения рекомендаций на основе контента - описания жанров.
    Матрица эмбеддингов размером (max_movie_id+1, n_genres), где строки
    соответствуют movieId, а столбцы — one-hot кодированию жанров.
    Матрица строится при инициализации экземпляра класса.
    """

    def __init__(self):
        self.embeddings = None
        self.ui_matrix = build_user_item_matrix()
        self._build_embeddings()

    def _build_embeddings(self):
        _, movies_df = load_data()
        self.movies_df = movies_df.copy()
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        vectorizer = CountVectorizer(tokenizer=lambda s: s.split("|"), lowercase=False)
        ###########################################################################
        # TODO: Строим матрицу эмбеддингов для фильмов и сохраняем в self.embeddings                       

        ###########################################################################
         # Создаём матрицу жанров
        genre_matrix = vectorizer.fit_transform(self.movies_df["genres"])
        
        # Получаем список жанров
        self.genre_names = vectorizer.get_feature_names_out()
        
        # Создаём эмбеддинги для всех фильмов
        # Нужно, чтобы индекс в матрице соответствовал movieId
        max_movie_id = max(self.movies_df["movieId"].max(), self.ui_matrix.shape[1]) + 1
        self.embeddings = np.zeros((max_movie_id, len(self.genre_names)))
        
        # Заполняем эмбеддинги для каждого фильма
        for idx, row in self.movies_df.iterrows():
            movie_id = row["movieId"]
            # Находим позицию фильма в genre_matrix
            # Используем toarray() для получения плотного вектора
            self.embeddings[movie_id] = genre_matrix[idx].toarray().flatten()

    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Предсказывает рейтинг user_id для item_id на основе контентной фильтрации.

        Алгоритм:
        1) Берём вектор целевого фильма: target_vec.
        2) Находим все фильмы, оцененные пользователем.
        3) Считаем косинусное сходство target_vec с векторами оцененных фильмов.
        4) Отбираем топ-k похожих оцененных фильмов (k-параметр).
        5) Предсказываем рейтинг как взвешенное среднее оценок по сходствам.
        6) Если не удаётся предсказать (нет оценок или нулевые векторы), возвращаем 0.0.
        7) Клипируем результат в [0.0, 5.0].

        Args:
            user_id: индекс пользователя
            item_id: индекс фильма
            k: сколько наиболее похожих оцененных фильмов использовать

        Returns:
            float: предсказанный рейтинг
        """
        # Вектор целевого фильма
        target_vec = self.embeddings[item_id].reshape(1, -1)
        
        # Находим все фильмы, оцененные пользователем
        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return 0.0
        
        # Берём векторы оцененных фильмов
        rated_vectors = self.embeddings[rated_items]
        
        # Проверяем, что векторы не нулевые
        rated_norms = np.linalg.norm(rated_vectors, axis=1)
        if np.sum(rated_norms) == 0:
            return 0.0
        
        # Считаем косинусное сходство
        similarities = cosine_similarity(target_vec, rated_vectors)[0]
        
        # Находим топ-k похожих
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        # Берём только те, у кого сходство > 0
        top_k_indices = [i for i in top_k_indices if similarities[i] > 0]
        
        if len(top_k_indices) == 0:
            return 0.0
        
        # Взвешенное среднее
        top_similarities = similarities[top_k_indices]
        top_ratings = user_ratings[rated_items[top_k_indices]]
        
        if np.sum(top_similarities) == 0:
            return 0.0
        
        predicted = np.sum(top_similarities * top_ratings) / np.sum(top_similarities)
        
        # Клипируем в [0, 5]
        return np.clip(predicted, 0.0, 5.0)
    
    def predict_items_for_user(
        self, user_id: int, k: int = 5, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы пользователю user_id на основе контента фильма.

        Алгоритм:
        1) Берем все фильмы, которые оценил пользователь.
        3) Строим профиль пользователя как взвешенное среднее жанров оцененных фильмов.
        4) Для всех фильмов, которые пользователь не оценил, считаем сходство с профилем.
        5) Сортируем по убыванию сходства и возвращаем top-n.
        """
        # Находим все фильмы, оцененные пользователем
        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return []
        
        # Берём векторы оцененных фильмов
        rated_vectors = self.embeddings[rated_items]
        ratings = user_ratings[rated_items]
        
        # Строим профиль пользователя (взвешенное среднее)
        # Нормализуем веса
        total_rating = np.sum(ratings)
        if total_rating > 0:
            weights = ratings / total_rating
        else:
            weights = np.ones(len(ratings)) / len(ratings)
        
        user_profile = np.sum(rated_vectors.T * weights, axis=1)
        user_profile = user_profile.reshape(1, -1)
        
        # Находим все фильмы, которые пользователь НЕ оценил
        all_items = np.arange(self.ui_matrix.shape[1])
        unrated_items = all_items[user_ratings == 0]
        
        if len(unrated_items) == 0:
            return []
        
        # Берём векторы неоцененных фильмов
        unrated_vectors = self.embeddings[unrated_items]
        
        # Считаем сходство с профилем пользователя
        similarities = cosine_similarity(user_profile, unrated_vectors)[0]
        
        # Сортируем по убыванию сходства
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Берём top-n рекомендаций
        recommendations = []
        for idx in sorted_indices:
            if len(recommendations) >= n_recommendations:
                break
            movie_id = unrated_items[idx]
            recommendations.append(int(movie_id))
        
        return recommendations


# Пример использования для дебага:
if __name__ == "__main__":
    user_id = 10
    item_id = 2
    k = 5
    content_recommender = ContentRecommender()
    print_user_rated_items(user_id, content_recommender.ui_matrix)

    pred_rating = content_recommender.predict_rating(user_id, item_id, k)
    print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating:.2f}")

    recommendations = content_recommender.predict_items_for_user(
        user_id, k=5, n_recommendations=10
    )
    for rec in recommendations:
        print(f"Recommended movie ID: {rec}, Title: {id_to_movie(rec)}")
