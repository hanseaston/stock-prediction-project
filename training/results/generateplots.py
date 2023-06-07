record_results(
        model_results_directory, "train_accuracy.csv", train_accuracy_hist, ["epoch", "accuracy", "num_predictions"], True)
    record_results(
        model_results_directory, "validation_accuracy.csv", validation_accuracy_hist, ["epoch", "accuracy", "num_predictions"])
    record_results(
        model_results_directory, "test_accuracy_50.csv", test_accuracy_hist_50, ["epoch", "accuracy", "num_predictions"])
    record_results(
        model_results_directory, "test_accuracy_60.csv", test_accuracy_hist_60, ["epoch", "accuracy", "num_predictions"])
    record_results(
        model_results_directory, "test_accuracy_70.csv", test_accuracy_hist_70, ["epoch", "accuracy", "num_predictions"])
    record_results(
        model_results_directory, "test_accuracy_80.csv", test_accuracy_hist_80, ["epoch", "accuracy", "num_predictions"])
    record_results(
        model_results_directory, "test_accuracy_90.csv", test_accuracy_hist_90, ["epoch", "accuracy", "num_predictions"])

    record_results(
        model_results_directory, "train_loss.csv", train_loss_hist, ["epoch", "loss"])
    record_results(
        model_results_directory, "validation_loss.csv", validation_loss_hist, ["epoch", "loss"])

    _, ax = plt.subplots()

    file_path = os.path.join(model_results_directory, "accuracy.png")
    x = get_data(train_accuracy_hist, 0)  # epoch num
    train_y = get_data(train_accuracy_hist, 1)
    valid_y = get_data(validation_accuracy_hist, 1)
    test_y_50 = get_data(test_accuracy_hist_50, 1)
    test_y_60 = get_data(test_accuracy_hist_60, 1)
    test_y_70 = get_data(test_accuracy_hist_70, 1)
    test_y_80 = get_data(test_accuracy_hist_80, 1)
    test_y_90 = get_data(test_accuracy_hist_90, 1)

    ax.plot(x, train_y, label='train')
    ax.plot(x, valid_y, label='validation')
    ax.plot(x, test_y_50, label='test_50')
    ax.plot(x, test_y_60, label='test_60')
    ax.plot(x, test_y_70, label='test_70')
    ax.plot(x, test_y_80, label='test_80')
    ax.plot(x, test_y_90, label='test_90')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.legend()
    plt.savefig(file_path)

    plt.clf()  # clean the plot for the next result
    _, ax = plt.subplots()

    file_path = os.path.join(model_results_directory, "loss.png")
    x = get_data(train_loss_hist, 0)  # epoch num
    train_y = get_data(train_loss_hist, 1)
    valid_y = get_data(validation_loss_hist, 1)
    ax.plot(x, train_y, label='train')
    ax.plot(x, valid_y, label='validation')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    plt.savefig(file_path)