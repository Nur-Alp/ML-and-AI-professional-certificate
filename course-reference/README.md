# Course reference material

This folder holds supplementary course materials unrelated to the main BBO
capstone project (see the repository root `README.md` for that). It exists
purely as personal reference and is kept clearly separate so it doesn't
dilute the capstone project's own documentation.

## module-14-svm/

Instructor-provided solution notebooks and datasets for Module 14 (Support
Vector Machines):

- `Self_study_try_it_14_1_solution/` — selecting hyperplanes for 2D data
- `Self_study_try_it_14_2_solution/` — training with different kernels on 2D data
- `Self_study_try_it_14_3_solution/` — soft-margin SVMs on 2D data
- `RequiredAssignment_14_1_solution/` — applying SVMs in Python (MNIST digit subsample)

`RequiredAssignment_14_1_solution/data/` is not stored in this repository,
it originally contained a standard MNIST digit subsample (`mnist_subsample_train.csv`,
`mnist_subsample_test.csv`), a large, publicly available dataset with no
need to be duplicated here. The full MNIST dataset is available from
[Yann LeCun's MNIST database](http://yann.lecun.com/exdb/mnist/) or via
`sklearn.datasets.fetch_openml('mnist_784')`.
