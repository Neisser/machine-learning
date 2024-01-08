use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

/// Represents a dataset containing input and output values.
///
/// This structure is used to store and manage collections of paired input and
/// output data points, typically for tasks like model training and evaluation.
#[derive(Debug)] // Enable printing for debugging
pub struct DataSet {
    /// A vector storing the input values of the dataset.
    pub input: Vec<f64>,
    /// A vector storing the corresponding output values of the dataset.
    pub output: Vec<f64>,
    /// The number of data points in the dataset.
    pub size: usize,
}

impl DataSet {
    /// Creates a new DataSet from a file.
    ///
    /// Reads lines from the specified file, parses them into input and output
    /// values, and constructs a DataSet instance. Handles potential errors during
    /// file reading and parsing.
    ///
    /// # Arguments
    ///
    /// * `path`: A Path object representing the file to read from.
    ///
    /// # Returns
    ///
    /// A DataSet containing the parsed input and output data. Returns an empty
    /// DataSet if file reading fails or the file is empty.
    pub fn new(path: &Path) -> DataSet {
        // Attempt to read lines from the file.
        if let Ok(lines) = DataSet::read_lines(path) {
            // Initialize vectors to store parsed data.
            let mut input_vec: Vec<f64> = Vec::new();
            let mut output_vec: Vec<f64> = Vec::new();

            // Iterate through each line and parse it.
            for line in lines {
                if let Ok(ip) = line {
                    // Split the line into comma-separated values.
                    let line: Vec<&str> = ip.split(",").collect();

                    // Parse the output value (first element).
                    let output: f64 = match line[0].parse() {
                        Ok(num) => num,
                        Err(_) => continue, // Skip invalid lines
                    };

                    // Parse the input value (second element).
                    let input: f64 = match line[1].parse() {
                        Ok(num) => num,
                        Err(_) => continue, // Skip invalid lines
                    };

                    // Add parsed values to the vectors.
                    input_vec.push(input);
                    output_vec.push(output);
                }
            }

            // Return a DataSet with the parsed data.
            return DataSet {
                size: input_vec.len(),
                input: input_vec,
                output: output_vec,
            };
        }

        // Return an empty DataSet if file reading fails.
        return DataSet {
            input: Vec::new(),
            output: Vec::new(),
            size: 0,
        };
    }

    /// Adds a new data point to the DataSet.
    ///
    /// Appends the provided input and output values to the respective vectors
    /// within the DataSet and increments the size counter.
    ///
    /// # Arguments
    ///
    /// * `input`: The input value to add to the DataSet.
    /// * `output`: The corresponding output value to add.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut dataset = DataSet::new("data.csv");
    /// dataset.add_row(2.5, 4.1); // Add a new data point
    /// ```
    #[allow(dead_code)]
    fn add_row(&mut self, input: f64, output: f64) {
        self.input.push(input);
        self.output.push(output);
        self.size += 1;
    }

    /// Reads lines from a file, handling potential errors.
    ///
    /// Opens the specified file, wraps it in a buffered reader, and returns an
    /// iterator over the lines of the file. Uses a `Result` type to handle errors
    /// that might occur during file opening.
    ///
    /// # Arguments
    ///
    /// * `filename`: A path-like object representing the file to read.
    ///
    /// # Returns
    ///
    /// An `io::Result` containing either:
    ///   * An `io::Lines` iterator over the lines of the file if successful.
    ///   * An `io::Error` indicating the reason for failure.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let lines = read_lines("data.txt").expect("Failed to read lines");
    /// for line in lines {
    ///     Process each line
    /// }
    /// ```
    fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
    where
        P: AsRef<Path>,
    {
        // Open the file using `File::open()` and handle potential errors.
        let file = File::open(filename)?;

        // Create a buffered reader for efficient line-by-line reading.
        Ok(io::BufReader::new(file).lines())
    }

    /// Normalizes the input and output values of a `DataSet` by subtracting the mean and dividing by the standard deviation.
    ///
    /// # Arguments
    ///
    /// * `self`: A mutable reference to the `DataSet` object.
    ///
    /// # Example usage
    ///
    /// ```rust
    /// let mut data = DataSet::new(...);
    /// data.normalize();
    /// ```
    pub fn normalize(&mut self) {
        // Calculate the mean of the input and output values.
        let input_mean: f64 = self.input.iter().sum::<f64>() / self.size as f64;
        let output_mean: f64 = self.output.iter().sum::<f64>() / self.size as f64;

        // Calculate the standard deviation of the input and output values.
        let input_std_dev: f64 = self
            .input
            .iter()
            .map(|x: &f64| (x - input_mean).powf(2.0))
            .sum::<f64>()
            .sqrt()
            / self.size as f64;
        let output_std_dev: f64 = self
            .output
            .iter()
            .map(|x: &f64| (x - output_mean).powf(2.0))
            .sum::<f64>()
            .sqrt()
            / self.size as f64;

        // Normalize the input and output values.
        self.input = self
            .input
            .iter()
            .map(|x: &f64| (x - input_mean) / input_std_dev)
            .collect();
        self.output = self
            .output
            .iter()
            .map(|x: &f64| (x - output_mean) / output_std_dev)
            .collect();
    }

}

