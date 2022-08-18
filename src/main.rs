use csv::ReaderBuilder;
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{prelude::*, OwnedRepr};
use ndarray_csv::Array2Reader;
use plotlib::{
    grid::Grid,
    page::Page,
    repr::Plot,
    style::{PointMarker, PointStyle},
    view::{ContinuousView, View}
};

fn main() {
    let train = load_csv("data/train.csv");
    let test = load_csv("data/test.csv");

    let features = train.nfeatures();
    let targets = train.ntargets();

    println!(
        "training with {} sample, testing with {} sample, {} features and {} targets",
        train.nsamples(),
        test.nsamples(),
        features,
        targets,
    );

    println!("plot data..");
    plot(&train);

    println!("training and testing model..");
    let mut max_confusion_matrix = iterate_value(&train, &test, 0.01, 100);
    let mut best_threshold = 0.0;
    let mut best_max_iterations = 0;
    let mut threshold = 0.02;

    for max_iteration in (1000..5000).step_by(500) {
        while threshold < 1.0 {
            let confusion_matrix = iterate_value(&train, &test, threshold, max_iteration);

            if confusion_matrix.accuracy() > max_confusion_matrix.accuracy() {
                max_confusion_matrix = confusion_matrix;
                best_threshold = threshold;
                best_max_iterations = max_iteration;
            }
            threshold += 0.01;
        }
        threshold += 0.02;
    }

    println!("most accurate confusion matrix: {:?}", max_confusion_matrix);
    println!("with max_iteration: {}, threshold: {}", best_max_iterations, best_threshold);
    println!("accuracy {}", max_confusion_matrix.accuracy());
    println!("precision {}", max_confusion_matrix.precision());
    println!("recall {}", max_confusion_matrix.recall());
}


fn load_csv(path: &str) -> Dataset<f64, &'static str> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_path(path)
        .expect("Couldn't load CSV file");

    let array: Array2<f64> = reader
        .deserialize_array2_dynamic()
        .expect("Couldn't deserialize CSV file");

    let data = array.slice(s![.., 0..2]).to_owned();
    let targets = array.column(2).to_owned();

    let feature_names = vec!["test 1", "test 2"];

    Dataset::new(data, targets)
        .map_targets(|x| {
            if *x as usize == 1 {
                "accepted"
            } else {
                "denied"
            }
        })
        .with_feature_names(feature_names)
}


fn iterate_value(
    train: &DatasetBase<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,
    >,
    test: &DatasetBase<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,
    >,
    threshold: f64,
    max_iterations: u64,
) -> ConfusionMatrix<&'static str> {
        let model = LogisticRegression::default()
            .max_iterations(max_iterations)
            .gradient_tolerance(0.0001)
            .fit(train)
            .expect("Couldn't train model");

        let validation = model.set_threshold(threshold).predict(test);

        let confusion_matrix = validation
            .confusion_matrix(test)
            .expect("Couldn't predict confusion matrix");

        confusion_matrix
    }


fn plot(train:
    &DatasetBase<
        ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
        ArrayBase<OwnedRepr<&'static str>, Dim<[usize; 2]>>,
    >) {
        let mut positive = Vec::new();
        let mut negative = Vec::new();

        let records = train.records().clone().into_raw_vec();
        let features: Vec<&[f64]> = records.chunks(2).collect();
        let targets = train.targets().clone().into_raw_vec();
        for i in 0..features.len() {
            let feature = features.get(i).expect("feature not exists");
            if let Some(&"accepted") = targets.get(i) {
                positive.push((feature[0], feature[1]));
            } else {
                negative.push((feature[0], feature[1]));
            }
        }

        let plot_positive = Plot::new(positive)
            .point_style(
                PointStyle::new()
                    .size(2.0)
                    .marker(PointMarker::Square)
                    .colour("#00ff00"),
            )
            .legend("Exam Results".to_string());

        let plot_negative = Plot::new(negative)
            .point_style(
                PointStyle::new()
                    .size(2.0)
                    .marker(PointMarker::Circle)
                    .colour("#0000ff"),
            );

        let grid = Grid::new(0, 0);

        let mut image = ContinuousView::new()
            .add(plot_positive)
            .add(plot_negative)
            .x_range(0.0, 120.0)
            .y_range(0.0, 120.0)
            .x_label("Test 1")
            .y_label("Test 2");

        image.add_grid(grid);

        Page::single(&image)
            .save("plot.svg")
            .expect("Couldn't save plot data");

    }