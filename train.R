library(mlflow)
library(glmnet)
library(magrittr)
library(keras)
mlflow_start_run()
train_file <- mlflow_param("train", type = "string", default = "~/dev/spark_summit/r-project/train.csv")
test_file <- mlflow_param("test", type = "string", default = "~/dev/spark_summit/r-project/test.csv")

epochs <- mlflow_param("epochs", default=100, type="integer")
train <- read.csv(train_file)
train_summary <- summary(train)
X <- model.matrix(price ~ ., data=train)[,2:15]
y <- train$price
write.table(train_summary, "train_summary.csv")
mlflow_log_artifact("train_summary.csv", artifact_path = "train_summary")
test <- read.csv(test_file)
test_summary <- summary(test)
write.table(test_summary, "test_summary.csv")
mlflow_log_artifact("test_summary.csv", artifact_path = "test_summary")
X_test <- model.matrix(price ~ ., data=test)[,2:15]
y_test <- test$price
model <- keras_model_sequential() 
MLflowLogger <- R6::R6Class("MLflowLogger",
                           inherit = KerasCallback,
                           public = list(
                             losses = NULL,
                             on_epoch_end = function(batch, logs = list()) {
                               mlflow_log_metric("rmse",logs[["loss"]]^.5)
                               mlflow_log_metric("val_rmse",logs[["val_loss"]]^.5)
                             }
                           ))
mlflow_logger <- MLflowLogger$new()
model %>%
  layer_dense(units = 32, activation = "relu", input_shape = dim(X)[2]) %>%
  #layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")
model %>% compile(
  loss = "mse",
  optimizer = optimizer_rmsprop()
)
history <- model %>% fit(x = X, y = y, epochs = epochs, validation_data = list(X_test, y_test), callbacks = list(mlflow_logger))
png("history.png")
plot(history)
dev.off()
mlflow_log_artifact("history.png", artifact_path = "history")
mlflow_log_model(model, artifact_path = "keras_model")
loss = (model %>% evaluate(X_test, y_test))^.5
print(paste("loss", loss))
yhat = model %>% predict(X_test)
resid = yhat - y_test
residual_df = data.frame(list(actual=y_test, prediction=yhat))
png("residual_distribution.png")
plot(density(resid))
dev.off()
png("residual_actual.png")
residual_df = data.frame(list(actual=y_test, residualn=resid))
plot(residual_df[order(residual_df$actual),])
dev.off()
mlflow_log_artifact("residual_distribution.png", artifact_path = "residual_distribution")
mlflow_log_artifact("residual_actual.png", artifact_path = "residual_actual")
mlflow_end_run()

