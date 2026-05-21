from data_loader import load_normalized_data
from tqdm import tqdm

class KNN:
	def __init__(self, n_neighbors):
		self.n_neighbors = n_neighbors


	def fit(self, X_normalized, Y):
		self.X_normalized = X_normalized
		self.Y = Y
		self.set_of_labels = set(Y)
		
		label_occurences = {label : 0 for label in self.set_of_labels}
		for y_value in Y:
			label_occurences[y_value] += 1
		
		max_occurences = max(label_occurences.values())

		
		self.label_weight = {label : max_occurences / label_occurences[label] \
									for label in self.set_of_labels}



	def predict(self, x_to_forcast):
		if not hasattr(self, "X_normalized") or not hasattr(self, "Y"):
			raise Exception("The model is not fit")

		distances = {}
		for index_x, current_x in tqdm(enumerate(self.X_normalized), total=len(self.X_normalized)):
			current_distance = KNN.euclidian_distance(current_x, x_to_forcast)
			distances[index_x] = float(f"{current_distance:.2f}")


		sorted_distances = KNN.sorted_dict_by_values(distances)

		indexes_nearest_neighbors = list(sorted_distances.keys())[: self.n_neighbors]


		label_counter = {label : 0 for label in self.set_of_labels}
		for index_point in indexes_nearest_neighbors:
			label = self.Y[index_point]
			label_counter[label] += self.label_weight[label]

		sorted_label_counter = KNN.sorted_dict_by_values(label_counter)
		predicted_label = list(sorted_label_counter.keys())[-1]
		
		return predicted_label



	@staticmethod
	def euclidian_distance(x1, x2):
		x1, x2 = list(x1), list(x2)[0]

		return sum([(a-b)**2 for a, b in zip(x1, x2, strict=True)])**(1/2)


	@staticmethod
	def sorted_dict_by_values(dict_object):
		sorted_dict = dict(sorted(dict_object.items(), key=lambda item: item[1]))
		return sorted_dict





if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
	knn_object = KNN(n_neighbors=7)
	knn_object.fit(X_normalized, Y)
	print(knn_object.predict([X_normalized[7]]))
	print(Y[7])


