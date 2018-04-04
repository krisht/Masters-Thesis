def validate(epoch):
     inputs, classes = get_sample(size=100, validation=True)
     vector_inputs = sess.run(inference_model, feed_dict={inference_input: inputs})
     del inputs

     tempClassifier = neighbors.KNeighborsClassifier(31)
     tempClassifier.fit(vector_inputs, classes)

     # All data (Files with Seizures & Files without Seizures)

     val_inputs, val_classes = get_sample(size=validation_size)
     vector_val_inputs = sess.run(inference_model, feed_dict={inference_input: val_inputs})
     del val_inputs

     pred_class = tempClassifier.predict(vector_val_inputs)

     percentage = len([i for i, j in zip(val_classes, pred_class) if i == j]) * 100.0 / validation_size

     if DEBUG:
          print("Validation Results: %.3f%% of of %d correct" % (percentage, validation_size))

     val_classes = list(map(lambda x: num_to_class[x], val_classes))
     pred_class = list(map(lambda x: num_to_class[x], pred_class))
     class_labels = [0, 1, 2, 3, 4, 5]
     class_labels = list(map(lambda x: num_to_class[x], class_labels))
     conf_matrix = confusion_matrix(val_classes, pred_class, labels=class_labels)
     np.set_printoptions(precision=2)

     np.save('./%s Results/%s_confusion_matrix_epoch%s_%.3f%%' % (curr_time, curr_time, epoch, percentage), conf_matrix)

     plot_confusion_matrix(conf_matrix, classes=class_labels, epoch=epoch, accuracy=percentage)

     compute_tSNE(vector_inputs, classes, epoch=epoch, accuracy=percentage, num_to_label=num_to_class)

     # Files with Seizures
     
     val_inputs_seizure, val_classes_seizure = get_sample(size=validation_size)
     vector_val_inputs_seizure = sess.run(inference_model, feed_dict={inference_input: val_inputs_seizure})
     del val_inputs_seizure

     pred_class_seizure = tempClassifier.predict(vector_val_inputs_seizure)

     percentage_seizure = len([i for i, j in zip(val_classes_seizure, pred_class_seizure) if i == j]) * 100.0 / validation_size

     if DEBUG:
          print("Validation Results: %.3f%% of of %d correct" % (percentage_seizure, validation_size))

     val_classes_seizure = list(map(lambda x: num_to_class[x], val_classes_seizure))
     pred_class_seizure = list(map(lambda x: num_to_class[x], pred_class_seizure))
     class_labels_seizure = [0, 1, 2, 3, 4, 5]
     class_labels_seizure = list(map(lambda x: num_to_class[x], class_labels_seizure))
     conf_matrix_seizure = confusion_matrix(val_classes_seizure, pred_class_seizure, labels=class_labels_seizure)
     np.set_printoptions(precision=2)

     np.save('./%s Results/%s_confusion_matrix_with_seizure_epoch%s_%.3f%%' % (curr_time, curr_time, epoch, percentage_seizure), conf_matrix_seizure)

     plot_confusion_matrix(conf_matrix_seizure, classes=class_labels_seizure, epoch=epoch, accuracy=percentage_seizure, with_seizure=True, title = "Confusion Matrix on Files with Seizure")

     #compute_tSNE(vector_inputs, classes, epoch=epoch, accuracy=percentage_seizure, num_to_label=num_to_class)

     # Files without Seizures
     
     val_inputs_without_seizure, val_classes_without_seizure = get_sample(size=validation_size)
     vector_val_inputs_without_seizure = sess.run(inference_model, feed_dict={inference_input: val_inputs_without_seizure})
     del val_inputs_without_seizure

     pred_class_without_seizure = tempClassifier.predict(vector_val_inputs_without_seizure)

     percentage_without_seizure = len([i for i, j in zip(val_classes_without_seizure, pred_class_without_seizure) if i == j]) * 100.0 / validation_size

     if DEBUG:
          print("Validation Results: %.3f%% of of %d correct" % (percentage_without_seizure, validation_size))

     val_classes_without_seizure = list(map(lambda x: num_to_class[x], val_classes_without_seizure))
     pred_class_without_seizure = list(map(lambda x: num_to_class[x], pred_class_without_seizure))
     class_labels_without_seizure = [0, 1, 2, 3, 4, 5]
     class_labels_without_seizure = list(map(lambda x: num_to_class[x], class_labels_without_seizure))
     conf_matrix_without_seizure = confusion_matrix(val_classes_without_seizure, pred_class_without_seizure, labels=class_labels_without_seizure)
     np.set_printoptions(precision=2)

     np.save('./%s Results/%s_confusion_matrix_without_seizure_epoch%s_%.3f%%' % (curr_time, curr_time, epoch, percentage_without_seizure), conf_matrix_without_seizure)

     plot_confusion_matrix(conf_matrix_without_seizure, classes=class_labels_without_seizure, epoch=epoch, accuracy=percentage_without_seizure, with_seizure=False, title = "Confusion Matrix on Files without Seizure")

     #compute_tSNE(vector_inputs, classes, epoch=epoch, accuracy=percentage_without_seizure, num_to_label=num_to_class)

     count_of_triplets = dict()

     return percentage, conf_matrix
