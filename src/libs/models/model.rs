/// A trait defining the core behaviors of a machine learning model.
///
/// This trait acts as a common interface for different model implementations,
/// ensuring they provide essential methods for data feeding, training, and
/// prediction.
pub trait Model<T, R> {
  /// Feeds input data to the model for processing.
  ///
  /// # Arguments
  ///
  /// * `data`: A slice of <T> values representing the input data.
  ///
  /// # Returns
  ///
  /// A `Result` indicating either success or an error that occurred during data feeding.
  // fn feed<T>(&mut self, data: T) -> Result<&mut Self, Box<dyn std::error::Error>>;
  

  /// Trains the model to learn patterns from the provided data.
  ///
  /// # Returns
  ///
  /// A `Result` indicating either success or an error that occurred during training.
  fn fit(&mut self) -> Result<(), Box<dyn std::error::Error>>;

  /// Makes predictions on new, unseen data using the trained model.
  ///
  /// # Arguments
  ///
  /// * `new_data`: A slice of f64 values representing the new data to make predictions on.
  ///
  /// # Returns
  ///
  /// A `Result` containing either:
  ///   * A `Vec<f64>` representing the model's predicted outputs for the new data.
  ///   * An error that occurred during prediction.
  fn predict(&mut self, value: T) -> Result<R, Box<dyn std::error::Error>>;
}