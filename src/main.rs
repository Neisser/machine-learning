mod libs;

use std::path::Path;

use libs::dataset::lineal_dataset::DataSet;

use libs::models::model::Model;

use libs::models::lineal_regression::{LinealRegression, LinealRegressionOptions};


fn main() {
    println!("Hello, world!");
    let filename: &Path = Path::new("./assets/lineal_dataset.csv");

    let mut training_data: DataSet = DataSet::new(filename);

    println!("dataset length {}", training_data.output.len());

    let _ = training_data.normalize();

    let mut lineal_regression: LinealRegression = LinealRegression::new(
        training_data,
        LinealRegressionOptions {
            epochs: 100,
            learning_rate: 0.01,
            normalize: false,
        },
    );

    let _ = lineal_regression.fit();

    println!("cost {}", lineal_regression.cost);
}
