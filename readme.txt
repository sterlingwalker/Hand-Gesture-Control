=============================================
************** Important  Info **************
=============================================
+-------------------------------------------+
+            Static Hand Gestures           |
+---------------------+---------------------+
|        palm         |    Switch Window    |
+---------------------+---------------------+
|        fist         |  Tab / Left Click*  |
+---------------------+---------------------+
|  like (thumbs up)   |      Enter Key      |
+---------------------+---------------------+
|dislike (thumbs down)|    Backspace Key    |
+---------------------+---------------------+
|         ok          | Space / Right Click*|
+---------------------+---------------------+
|        peace        |    Close Window     |
+---------------------+---------------------+
|    one (pointing)   | Toggle Cursor Mode  |
+---------------------+---------------------+
* In cursor mode: fist = left click, ok = right click

+-------------------------------------------+
+            Dynamic Swipe Gestures         |
+---------------------+---------------------+
|      swipe_up       |     Scroll Down     |
+---------------------+---------------------+
|     swipe_down      |      Scroll Up      |
+---------------------+---------------------+
|     swipe_left      |   Previous Window   |
+---------------------+---------------------+
|     swipe_right     |     Next Window     |
+---------------------+---------------------+

*** Runnable Scripts (scripts/) ***
demo_webcam.py          = Opens webcam and predicts gestures (preview only).
run_hgr.py              = Main application. Controls mouse/keyboard using hand gestures.
train_hagrid.py         = Train the model on HaGRID dataset.
eval_hagrid.py          = Evaluate trained model on test set.
prepare_hagrid_subset.py= Prepare HaGRID dataset for training.

*** Support Modules (src/hand_gesture_control/) ***
data.py                 = Dataset loading and image transforms.
model.py                = EfficientNet model and checkpoint save/load.
train_utils.py          = Training loop utilities.
inference.py            = Prediction smoothing and gesture state tracking.
actions.py              = UI action mapping (gestures to keyboard/mouse).

*** Colab (colab/) ***
Hand_Gesture_Control_colab.py = Google Colab training script.

=============================================
************* Environment Setup *************
=============================================

To start the Hand Gesture Recognition, it is highly recommended to set up a virtual python
environment using conda.

In your console:

# Create Environment
conda create -n gesture python=3.10 -y

# Activate it
conda activate gesture

# Install Dependencies
pip install -r requirements.txt



=============================================
************ Running The Program ************
=============================================
After you have activated the gesture environment

# To run the demo (preview only, no actions)
python scripts/demo_webcam.py
### controls ###
q = quit

# To run the Hand Gesture Recognition UI Controller
python scripts/run_hgr.py
### controls ###
q = quit
p = pause/resume actions
r = reset state

### optional flags ###
--no-actions    = Preview mode, gestures shown but no actions triggered
--confidence    = Set minimum confidence threshold (default: 0.90)
--hold-frames   = Frames to hold gesture before triggering (default: 10)

### safety ###
Move mouse to any screen corner to abort (pyautogui failsafe)



