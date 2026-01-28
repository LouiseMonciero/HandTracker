import os
import glob
import argparse
import json
from tkinter import *
from PIL import Image, ImageTk


# Labels decoder ------------------
LABELS_MAPPING = {
    "heart_1": 0,
    "heart_2": 1,
    "heart_3": 2,
    "index_lifted": 3,
    "f*ck_sign": 4,
    "victory_sign": 5,
    "ok": 6,
    "fist": 7,
    "middle_finger_touching_thumb": 8,
    "little_finger_up": 9,
    "ring_finger_touching_thumb": 10,
    "triangle": 11,
    "thmubs_up":12,
}
LABELS_MAPPING_REV = {v:k for k, v in LABELS_MAPPING.items()}

def generate_label(label_file, image_dir):

    def first_image():
        """Display the first image and initialize the label states variables"""
        nonlocal current_img_idx
        global current_label, current_hand_selected
        if current_img_idx < len(path):
            img_path = path[current_img_idx]
            current_label = label_data['labels'][current_img_idx]['label']
            current_hand_selected = None
            if type(current_label) is list:
                current_hand_selected = 3 if len(current_label)==1 else None

            update_current_label_text()
            display_image(img_path)

    def next_image():
        """Actualize states variable current_img_idx and the label states variables"""
        nonlocal current_img_idx
        global current_label, current_hand_selected
        if current_img_idx < len(path):
            current_img_idx += 1
            img_path = path[current_img_idx]
            current_label = label_data['labels'][current_img_idx]['label']
            current_hand_selected = None
            if type(current_label) is list:
                current_hand_selected = 3 if len(current_label)==1 else None

            update_current_label_text()
            display_image(img_path)
            
        else:
            print("Toutes les images ont été étiquetées.")
            #root.quit()
    
    def prev_image():
        """Actualize states variable current_img_idx and the label states variables"""
        nonlocal current_img_idx
        global current_label, current_hand_selected
        if current_img_idx > 0:
            current_img_idx -= 1
            current_label = label_data['labels'][current_img_idx]['label']
            current_hand_selected = None
            if type(current_label) is list:
                current_hand_selected = 3 if len(current_label)==1 else None
            
            img_path = path[current_img_idx]
            update_current_label_text()
            display_image(img_path)
        else:
            print("No previous image")

    def display_image(img_path):

        # Load image
        img = Image.open(img_path)
        img = img.resize((480, 270))
        img_tk = ImageTk.PhotoImage(img)

        # Add image to the canvas
        canvas.create_image(0, 0, anchor=NW, image=img_tk)
        canvas.image = img_tk

    def assign_current_hand_selected(hand_selected):
        global current_hand_selected, current_label

        if current_hand_selected == hand_selected:
            return
        
        current_hand_selected = hand_selected

        if type(current_label) is list: # jsp
            if len(current_label) ==2 and hand_selected ==3:
                current_label = [None]
                label_data['labels'][current_img_idx]['label'] = current_label
            elif len(current_label) ==1 and hand_selected !=3:
                current_label = [None, None]
                label_data['labels'][current_img_idx]['label'] = current_label
        else :
            if current_hand_selected == 1: #hand 1
                current_label = [current_label, None]
            elif current_hand_selected == 2: #hand 2
                current_label = [None, current_label]
            elif current_hand_selected == 3: # both hands
                current_label = [current_label]
            label_data['labels'][current_img_idx]['label'] = current_label
        update_current_label_text()
    
    def assign_current_label(label):
        global current_label
        global label_data
        global current_hand_selected
        if current_hand_selected is None: # default
            current_label = label
        elif current_hand_selected==1 : #hand 1
            current_label[0] = label
            label_data['labels'][current_img_idx]['label'] = current_label
        elif current_hand_selected==2: # hand 2
            current_label[1] = label
            label_data['labels'][current_img_idx]['label'] = current_label
        elif current_hand_selected==3: # both hands
            current_label = [label]
            label_data['labels'][current_img_idx]['label'] = current_label
        update_current_label_text()
        print("current label changed to :",current_label)

    def update_current_label_text():
        global current_label, current_hand_selected
        if current_hand_selected == 3: # both hands
            current_label_var.set(
                f"Both Hands :{current_label if current_label is None else ( 'None' if current_label[0] is None else LABELS_MAPPING_REV[current_label[0]])}"
            )
        
        else :
            current_label_var.set(
                f"Hand 1:{current_label if current_label is None else ( 'None' if current_label[0] is None else LABELS_MAPPING_REV[current_label[0]])}\
                Hand 2: {current_label if current_label is None else ('None' if current_label[1] is None else LABELS_MAPPING_REV[current_label[1]])}")
    
    def save_label_data(label_file):
        
        with open(label_file, 'w') as f:
            json.dump(label_data, f, indent=4)
        print('saved !')

    def load_json(label_file):
        global path
        global label_data

        # Creating the file if doesn't exist
        if not os.path.exists(label_file):
            with open(label_file, 'w') as f:
                json.dump({"labels": []}, f)

        path = glob.glob(os.path.join(image_dir, '*'))

        with open(label_file, 'r') as f:
            label_data = json.load(f)
        
        # Add every image to the json if not already present
        for img in path:
            if not any(label['path'] == img for label in label_data['labels']):
                label_data['labels'].append({
                    "path": img,
                    "label": None,
                })

        # Save label_data
        save_label_data(label_file)

    load_json(label_file)

    current_label = [None, None]
    current_hand_selected = None
    
    # Mise en page TKINTER ------------------

    root = Tk()
    root.title("Image Labeling Tool")

    current_img_idx = 0

    # Ajout des boutons pour les étiquettes
    label_var = StringVar()
    label_var.set(None)

    # Text for label
    current_label_var = StringVar()
    current_label_var.set(f"Actual label : {current_label}")

    button_frame = Frame(root)
    button_frame.pack()

    # Left buttons
    left_frame = Frame(root)
    left_frame.pack(side="left", padx=10, pady=10)
    buttons = []
    for label in LABELS_MAPPING:
        # Note : buttons has same index than label they store.
        buttons.append(Button(left_frame, text=label, command=lambda label=LABELS_MAPPING[label]: assign_current_label(label)))
        buttons[len(buttons) -1].pack(side="top", fill="x", padx=5, pady=5)

    # Image
    middle_frame = Frame(root)
    middle_frame.pack(padx=10, pady=10) # not sure
    canvas = Canvas(middle_frame, width=480, height=270)
    canvas.pack()

    # Right buttons
    right_frame = Frame(root)
    right_frame.pack(side="right", padx=10, pady=10)
    b_hands_1 = Button(right_frame, text="Hand 1", command=lambda: assign_current_hand_selected(1))
    b_hands_2 = Button(right_frame, text="Hand 2", command=lambda: assign_current_hand_selected(2))
    b_hands_both = Button(right_frame, text="Both hands", command=lambda: assign_current_hand_selected(3))
    b_hands_1.pack(side=LEFT)
    b_hands_2.pack(side=LEFT)
    b_hands_both.pack(side=LEFT)

    # Bottom buttons - navigation
    bottom_frame = Frame(root)
    bottom_frame.pack(side="bottom", padx=10, pady=10)
    b_prev = Button(root, text="<-", command=prev_image).pack(side=LEFT)
    b_next = Button(root, text="->", command=next_image).pack(side=LEFT)
    b_save = Button(root, text="Save", command=lambda : save_label_data(label_file)).pack(side=LEFT)
    
    # Display first image
    first_image()

    # Bottom label text
    bottom_text_frame = Frame(root)
    bottom_text_frame.pack(side="bottom", pady=(0, 10))

    label_info = Label(
        bottom_text_frame,
        textvariable=current_label_var,
        fg="#790079",          # violet
        font=("Arial", 20, "bold")
    )
    label_info.pack()



    # Lancer la boucle Tkinter
    root.mainloop()

def main():
    ap = argparse.ArgumentParser(
        description="Label all the images of a directory"
    )
    ap.add_argument("--label-file", default='./data/labels.json', help="File where all labels are stored (eg. 'labels.json')")
    ap.add_argument("--image-dir", default='./data/images/', help="Directory of all images to label")
    args = ap.parse_args()

    generate_label(
        label_file=args.label_file,
        image_dir=args.image_dir
    )

if __name__ == "__main__":
    main()
