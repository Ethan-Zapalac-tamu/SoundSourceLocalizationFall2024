import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os

# machine learning ssl method
from sklearn import linear_model
import pickle

# loading bar
from tqdm import tqdm


def hardware_volume_ssl(max_volume_values, ambient_volume = 0, debug_prints=False):
    # the input is in format [[amplitude0, time0], [amplitude1, time1]...]
    # returns an estimated direction of the sound source in 3d space
    if debug_prints:
        print(f"Debug volumes:{max_volume_values}")
    directions = [0.0] * 3
    # pick whatever direction is the loudest, go from there
    loudest_amplitude = 0
    for mic in max_volume_values:
        if mic[0] > loudest_amplitude:
            loudest_amplitude = mic[0]
    if debug_prints:
        print(f"Loudest ampl:{loudest_amplitude}")
    # x direction is from front to back
    # mic3 is positive, mic0 is negative
    pos_x = max_volume_values[3][0] / loudest_amplitude
    neg_x = max_volume_values[0][0] / loudest_amplitude
    directions[0] = (pos_x - neg_x)
    # y direction is from white side to blue side
    # mic4 is positive and mic2 is negative
    pos_y = max_volume_values[4][0] / loudest_amplitude
    neg_y = max_volume_values[2][0] / loudest_amplitude
    directions[1] = (pos_y - neg_y)
    # only z axis is upwards mic, mic1
    # calc z based on how loud it should be
    directions[2] = ((max_volume_values[1][0] / loudest_amplitude) - 0.5)*2

    return directions

def volume_based_ssl(max_volume_values):
    print("Volume based SSL")
    # assumes mics are arranged orthogonally and facing outwards
    directions = [0.0] * len(max_volume_values)
    # Assume initially the sound is in the loudest direction

    loudest_sound = 0
    loudest_mic_i = 0
    for volume_i in range(len(max_volume_values)):
        amp_of_volume = max_volume_values[volume_i][0]
        print(f"amplitude of mic_{volume_i} = {amp_of_volume}")
        if amp_of_volume > loudest_sound:
            loudest_sound = amp_of_volume
            loudest_mic_i = volume_i

    for mic_reading_i in range(len(max_volume_values)):  # Fixing the range to use len()
        # compare volume and assume direction
        amp_of_volume = max_volume_values[mic_reading_i][0]
        directions[mic_reading_i] = amp_of_volume / loudest_sound
        # change range of values, so about 20% of volume is considered behind
        directions[mic_reading_i] = (((directions[mic_reading_i] - 0.2) / 0.8)-0.5)/0.5

    # predict direction based on how far back the sound is expected to be
    print("Relative to the position of Mic", loudest_mic_i)

    return directions


def generate_model(data_points_x, expected_outputs_y):
    ols_model = linear_model.LinearRegression()
    ols_model.fit(data_points_x, expected_outputs_y)
    print(f"regression coefficients = {ols_model.coef_}")
    # save model using pickle
    with open('ssl_model.pkl', 'wb') as f:
        pickle.dump(ols_model, f)

    return ols_model


def predict_with_model(input_data, ols_model_path="ssl_model.pkl"):
    # Load the model using pickle
    with open(ols_model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    # Use the model to predict the output for each data point
    predictions = loaded_model.predict(input_data)

    return predictions


def calc_unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    if np.linalg.norm(vector) == 0:
        return [0, 0, 0]
    return vector / np.linalg.norm(vector)

def angle_between_vectors(v1, v2):
    """ Returns the angle between 2 vectors in radians"""
    v1_unit = calc_unit_vector(v1)
    v2_unit = calc_unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0))


# def digital_filter(csv_file_data):
#     # we want to filter sounds outside 1000 Hz
#     # cutoff frequencies 900 Hz to 1100 Hz
#     num_taps = 31
#     cut_off = 1200 # Hz
#     sample_rate = 30 # Hz
#     # cutoff frequencies are given in kHz
#     # sample rate limits digital filter max cutoff to only 15 Hz
#     d_filter = firwin(num_taps, [900, 1200], pass_zero=False, fs=sample_rate)
#     print(d_filter)
#     return -1


# create a program to parse a csv file

# find the sound

# determine its location based on volume or time delay
def predict_direction(file_name, expected_direction=[0, 0, 0], debug=False):
    # print("in predict_direction")

    # Upwards mic = 1
    # Front mic = 3
    # Back mic = 0
    # Blue side mic = 2
    # White side mic = 4

    # frontside = [0, -1, 0]
    # backside = [0, 1, 0] check these
    # blue side = [1, 0, 0]
    # white side = [-1, 0, 0]
    # topside = [0, 0, 1]
    # down = [0, 0, -1]

    # open the csv file in read only mode
    with open(file_name, mode='r') as csv_file:
        if debug:
            print("opened file")
        # Create a CSV reader object
        csv_reader = pd.read_csv(csv_file)
        if debug:
            print(f"length = {pd.options.display.max_rows}")
            print(f"type of csv_reader={type(csv_reader)}")
        # there won't be headers for this

        num_microphones = 5 # hardware has 5 mics

        # iterate through rows of the csv file
        initialize_sound_i = 0

        # baseline array a list with the min and max values for the baseline
        # [[min0, max0], [min1, max1], ... [minh, maxh]]
        baseline = np.zeros((num_microphones, 2))
        baseline_amps = np.zeros(num_microphones)


        # keeps track of the loudest amplitude, and when it happens
        # [[amplitude1, time(line number),[...]]
        sound_locate_volume = np.zeros((num_microphones, 2))

        # previous 4 values for each mic to calculate amplitudes
        running_amplitude = np.zeros((num_microphones, 4))

        row_number = 0
        initialization_length = 20

        for row in csv_reader.itertuples():
            # assume that the startup of the file is the soundless baseline
            # get the "baseline sound" for each mic

            if initialize_sound_i <= initialization_length:
                initialize_sound_i += 1
                for i in range(num_microphones):
                    # check if the baseline has not been set yet
                    if baseline[i][0] == 0:
                        baseline[i][0] = int(row[i])
                    if baseline[i][1] == 0:
                        baseline[i][1] = int(row[i])
                    # find if it is the new min/max of the baseline
                    if int(row[i]) < int(baseline[i][0]):
                        baseline[i][0] = int(row[i])
                    if int(row[i]) > int(baseline[i][1]):
                        baseline[i][1] = int(row[i])
                if initialize_sound_i == initialization_length:
                    # load baseline into initial state for running amplitude
                    for j in range(num_microphones):
                        for k in range(len(running_amplitude[j])):
                            if k%2==0:
                                running_amplitude[j][k] = baseline[j][0]
                            else:
                                running_amplitude[j][k] = baseline[j][1]
                    # calculate baseline amplitudes
                    for j in range(num_microphones):
                        baseline_amps[j] = baseline[j][1] - baseline[j][0]



            else:
                # this is where the bulk of the data is processed
                # "detect sound if the current spot is significantly louder than the baseline"
                if len(row) == 6:
                    timestamp = row[5]
                else:
                    timestamp = row_number

                # TODO add either digital filter or frequency pattern recognition

                # frequency of sound source is 1000 Hz
                # attempt to only include sounds that repeat every 1000 hz

                for i in range(num_microphones):
                    # calc current amplitude based on absolute difference between previous signal value
                    # move old values down the line
                    # this is not replacing old values
                    temp_array = np.insert(running_amplitude[i], 0, row[i])
                    running_amplitude[i] = np.delete(temp_array, -1)

                    # calculate running amplitude
                    running_min = np.min(running_amplitude[i])
                    running_max = np.max(running_amplitude[i])
                    # subtract baseline amplitude
                    running_amp_value = (running_max - running_min) - baseline_amps[i]

                    # saving value if needed
                    if running_amp_value > sound_locate_volume[i][0]:
                        sound_locate_volume[i][0] = running_amp_value
                        sound_locate_volume[i][1] = timestamp


            row_number += 1
            # print(running_amplitude)
            # detect sound "louder than the baseline amplitude"
            # assuming that there is only 1 sound in the file, base direction on volume



        directions = hardware_volume_ssl(sound_locate_volume)
        if debug:
            print("Volume and time of loudest sound")
            print(sound_locate_volume)
            print("Estimated direction, 0 is completely behind that mic, 1 is completely in front of that mic")
            print(directions)
            print(file_name)
        # plot direction

        fig1 = plt.figure()  # Create a single figure
        ax1 = plt.axes(projection="3d")
        v1 = [directions[0], directions[1], directions[2]]
        # print(v1[2], v1[1], v1[0])
        # plotting multiple lines at once
        # plot the 2 vectors
        unit_expected = calc_unit_vector(expected_direction)
        ax1.quiver(0, 0, 0, v1[0], v1[1], v1[2], color="red")
        ax1.quiver(0, 0, 0, unit_expected[0], unit_expected[1], unit_expected[2], color="black")
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        angle_error = angle_between_vectors(v1, expected_direction)
        if debug:
            print(f"angle_error in rads={angle_error}, deg={math.degrees(angle_error)}")
            print(f"expected= {expected_direction}")

        plt.title(f"Angle between prediction and expected: {"{:.3f}".format(math.degrees(angle_error))}°")
        fig1.suptitle("Direction Estimation", fontsize=14, fontweight='bold')
        plt.legend(["Predicted", "Expected"], loc="lower right")
        # plt.show()
        save_fig_to_file = file_name[:-4] + "_plot_filter.png"
        plt.savefig(save_fig_to_file)
        # calculate the position of the sound source based on polar coordinates
        # close figure
        plt.close(fig1)
        return expected_direction, math.degrees(angle_error), sound_locate_volume

def main():
    # go through every .csv file in the folder and predict its direction
    # The expected direction should be the file name
    path = "filter_final"
    files_in_folder = os.listdir(path)
    errors = []

    train_model_data_x = []
    train_model_labels_y = []
    expected_direction_list = []

    band_aid_filename = 0
    print(f"Performing SSL on all files in {path}")
    for file in tqdm(files_in_folder):
        # only send the cleaned .csv files
        if ".csv" in file:
            # print(file)
            # get predicted direction from file name
            split_file_name = file.split(".") # should be <file_name>.csv
            expected_direction_text = split_file_name[0].split("_") # should be ["#", "#", "#"]
            # seems to ignore the final number in #_#_#_#.csv, letting us use multiple readings for the same data point
            expected_direction_vector = [int(expected_direction_text[0]), int(expected_direction_text[1]), int(expected_direction_text[2])]
            # save for training the model
            train_model_labels_y.append(calc_unit_vector(expected_direction_vector))
            data_name = expected_direction_vector.copy()
            data_name.append(band_aid_filename)
            expected_direction_list.append(data_name)
            band_aid_filename += 1
            # send to prediction
            relative_file_name = path + "/" + file
            angle, error, model_info = predict_direction(relative_file_name, expected_direction_vector)
            errors.append([angle, error])

            # format model training data
            add_model_data_point = []
            for amplitude in model_info:
                # print(f"AmpRaw={amplitude}")
                # print(f"AmpRaw0={amplitude[0]}")
                # print(f"AmpRaw1={amplitude[1]}")
                add_model_data_point.append(int(amplitude[0]))
            train_model_data_x.append(add_model_data_point)


    # print angle errors
    sum_of_errors = 0
    n_errors = 0
    for i in errors:
        for direction in i[0]:
            print(f"{int(direction)}, ", end="")
        print(i[1])
        sum_of_errors += i[1]
        n_errors+=1
    average_degrees_off = sum_of_errors / n_errors
    print(f"Average degrees off = {average_degrees_off}")

    # apply machine learning method

    # print("model data X")
    # for data_index in range(len(train_model_data_x)):
    #     print(f"X={train_model_data_x[data_index]}, y={train_model_labels_y[data_index]}")
    ols_model = generate_model(train_model_data_x, train_model_labels_y)

    # compare ML method with regular training method
    predictions_using_ols = predict_with_model(train_model_data_x)
    print(f"predictions = \n{predictions_using_ols}")
    print("errors in predictions using ols model")
    ols_model_errors = []
    ols_sum_of_errors = 0
    ols_num_errors = 0
    for data_i in range(len(predictions_using_ols)):
        ols_model_errors.append(angle_between_vectors(predictions_using_ols[data_i], train_model_labels_y[data_i]))
        # print(f"{train_model_labels_y[data_i]}, {math.degrees(ols_model_errors[data_i])}")
        ols_sum_of_errors += math.degrees(ols_model_errors[data_i])
        ols_num_errors += 1
        # plot the data using the machine learing method
        fig1 = plt.figure()  # Create a single figure
        ax1 = plt.axes(projection="3d")
        v1 = np.array([predictions_using_ols[data_i][0], predictions_using_ols[data_i][1], predictions_using_ols[data_i][2]])
        unit_expected = calc_unit_vector(train_model_labels_y[data_i])
        ax1.quiver(0, 0, 0, v1[0], v1[1], v1[2], color="red")
        ax1.quiver(0, 0, 0, unit_expected[0], unit_expected[1], unit_expected[2], color="black")
        ax1.set_xlim([-1, 1])
        ax1.set_ylim([-1, 1])
        ax1.set_zlim([-1, 1])

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        angle_error = angle_between_vectors(v1, np.array(unit_expected))
        debug = False
        if debug:
            print(f"v1={v1}")
            print(f"unit_expected={unit_expected}")
            print(f"angle_error in rads={angle_error}, deg={math.degrees(angle_error)}")
            print(f"expected= {unit_expected}")
        plt.title(f"Angle between prediction and expected using ML: {"{:.3f}".format(math.degrees(angle_error))}°")
        fig1.suptitle("Direction Estimation", fontsize=14, fontweight='bold')
        plt.legend(["Predicted", "Expected"], loc="lower right")
        # plt.show()
        save_fig_to_file = path + "/"
        for value in expected_direction_list[data_i]:
            save_fig_to_file += str(value) + "_"
        save_fig_to_file += "plot_finalH.png"
        plt.savefig(save_fig_to_file)
        # calculate the position of the sound source based on polar coordinates
        # close figure
        plt.close(fig1)



    # print average error
    ols_average_degrees_off = ols_sum_of_errors / ols_num_errors
    print(f"Average degrees off in ols model = {ols_average_degrees_off}")













main()