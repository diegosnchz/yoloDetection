import mediapipe as mp
print("Files in mp:", dir(mp))
try:
    import mediapipe.solutions
    print("Imported mediapipe.solutions")
    print("Solutions dir:", dir(mp.solutions))
except ImportError as e:
    print("Failed to import mediapipe.solutions:", e)

try:
    import mediapipe.python.solutions
    print("Imported mediapipe.python.solutions")
except ImportError as e:
    print("Failed to import mediapipe.python.solutions:", e)
