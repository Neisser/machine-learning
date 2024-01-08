use crate::libs::dataset::lineal_dataset::DataSet;

/// Calculates the mean squared error (MSE) of a linear regression model on a given dataset.

/// Args:
/// * `b`: The bias term of the model.
/// * `w`: The weight vector of the model.
/// * `data`: A DataSet object containing the input and output data points.

/// Returns:
/// The mean squared error of the model on the given dataset.

/// Example usage
/// ```rust
/// let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let output_data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let data = DataSet::new(input_data, output_data);
///
/// let b = 1.0;
/// let w = 2.0;
///
/// let cost = calculate_cost(&b, &w, &data);
///
/// println!("Mean squared error: {}", cost);
/// ```

pub fn calculate_cost(b: &f64, w: &f64, data: &DataSet) -> f64 {
    // Calculate squared errors for each data point
    let squared_errors: f64 = data
        .output
        .iter()
        .zip(data.input.iter())
        .map(|(x, y)| (b + (w * x) - y).powf(2.0))
        .sum();

    // Handle potential division by zero
    if data.size == 0 {
        return f64::NAN; // Or return a default value or panic, depending on your error handling strategy
    }

    // Calculate mean squared error
    return squared_errors / (2.0 * data.size as f64);
}

/// Performs gradient descent to optimize the bias and weight of a linear model.
///
/// Iterates through a specified number of iterations, adjusting the bias and weight
/// based on the calculated gradients to minimize the error on the given training data.
///
/// # Arguments
///
/// * `num_iterations`: The number of iterations to perform gradient descent.
/// * `learning_rate`: The step size used to update the bias and weight in each iteration.
/// * `initial_bias`: The initial value for the bias term.
/// * `initial_weight`: The initial value for the weight term.
/// * `training_data`: A `DataSet` containing the input and output data for training.
///
/// # Returns
///
/// A tuple containing the optimized bias and weight values.
///
/// # Example usage
///
/// ```rust
/// let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let output_data = vec![2.0, 4.0, 5.0, 4.0, 5.0];
/// let data = DataSet::new(input_data, output_data);
///
/// let initial_bias = 0.0;
/// let initial_weight = 0.0;
///
/// let (optimized_bias, optimized_weight) = gradient_descent(
///     100, // Number of iterations
///     0.01, // Learning rate
///     &initial_bias,
///     &initial_weight,
///     &data,
/// );
///
/// println!("Optimized bias: {}", optimized_bias);
/// println!("Optimized weight: {}", optimized_weight);
/// ```
pub fn gradient_descent(
    num_iterations: u32,
    learning_rate: f64,
    initial_bias: &f64,
    initial_weight: &f64,
    training_data: &DataSet,
) -> (f64, f64) {
    let mut bias: f64 = *initial_bias;
    let mut weight: f64 = *initial_weight;

    for _ in 0..num_iterations {
        let num_data_points: f64 = training_data.size as f64;

        let bias_gradient: f64 = training_data
            .output
            .iter()
            .zip(training_data.input.iter())
            .map(|(x, y)| bias + weight * x - y)
            .sum();

        let weight_gradient: f64 = training_data
            .output
            .iter()
            .zip(training_data.input.iter())
            .map(|(x, y)| (bias + weight * x - y) * x)
            .sum();

        bias -= learning_rate * bias_gradient / num_data_points;
        weight -= learning_rate * weight_gradient / num_data_points;
    }

    (bias, weight)
}
