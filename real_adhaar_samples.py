from PIL import Image, ImageDraw, ImageFont
import random
import os

# Paths
template_path = r"D:\empty1.png"    
output_folder = "generated_aadhaar_cards"


# Photo folders
male_photo_folder = r"males"
female_photo_folder = r"females"

# Font file paths
font_hindi_path = "Noto_Sans_Devanagari/static/NotoSansDevanagari-Regular.ttf"
font_english_path = "arial.ttf"
font_dob_path = "arial.ttf"
font_aadhaar_path = "courbd.ttf"


base_font_size = 22

os.makedirs(output_folder, exist_ok=True)


name_mapping = [
    ("Amit Sharma", "अमित शर्मा", "Male"),
    ("Ravi Patel", "रवि पटेल", "Male"),
    ("Sita Verma", "सीता वर्मा", "Female"),
    ("Rahul Gupta", "राहुल गुप्ता", "Male"),
    ("Pooja Reddy", "पूजा रेड्डी", "Female"),
    ("Vijay Mishra", "विजय मिश्रा", "Male"),
    ("Neha Yadav", "नेहा यादव", "Female"),
    ("Anjali Singh", "अंजलि सिंह", "Female"),
    ("Deepak Kumar", "दीपक कुमार", "Male"),
    ("Kavita Joshi", "कविता जोशी", "Female")
]

def random_aadhaar_number():
    """Generate random Aadhaar number in XXXX XXXX XXXX format."""
    return " ".join(["".join([str(random.randint(0, 9)) for _ in range(4)]) for _ in range(3)])

def random_dob():
    """Generate random date of birth in DD/MM/YYYY format."""
    return f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(1965, 2000)}"

def get_random_photo(gender):
    """Get a random photo from male or female folder and resize."""
    folder = male_photo_folder if gender == "Male" else female_photo_folder
    photo_list = os.listdir(folder)
    random_photo = random.choice(photo_list)
    photo = Image.open(os.path.join(folder, random_photo)).convert("RGB")
    photo = photo.resize((192, 210), Image.LANCZOS)  # Resize to Aadhaar photo dimensions
    return photo


for i in range(1, 101):

    card = Image.open(template_path).copy()
    draw = ImageDraw.Draw(card)

    w, h = card.size
    scale = w / 600 

   
    font_hindi_scaled = ImageFont.truetype(font_hindi_path, int(base_font_size * scale))
    font_english_scaled = ImageFont.truetype(font_english_path, int(base_font_size * scale))
    font_dob_scaled = ImageFont.truetype(font_dob_path, int((base_font_size - 3) * scale))
    font_aadhaar_scaled = ImageFont.truetype(font_aadhaar_path, int((base_font_size + 6) * scale))  # Aadhaar number bigger

    name_english, name_hindi, gender = random.choice(name_mapping)
    aadhaar_number = random_aadhaar_number()
    dob = random_dob()


    user_photo = get_random_photo(gender)
    photo_left = int(w * 0.055)  
    photo_top = int(h * 0.27)    
    photo_width = 192
    photo_height = 210
    card.paste(user_photo, (photo_left, photo_top))

    x_right_of_photo = photo_left + photo_width + 30  
    y_start = photo_top + 5
    line_spacing = int(h * 0.085)

    
    qr_left = int(w * 0.8)


    aadhaar_text_bbox = draw.textbbox((0, 0), aadhaar_number, font=font_aadhaar_scaled)
    aadhaar_text_width = aadhaar_text_bbox[2] - aadhaar_text_bbox[0]
    aadhaar_text_height = aadhaar_text_bbox[3] - aadhaar_text_bbox[1]

    
    available_space_left = photo_left + photo_width + 10
    available_space_right = qr_left - 10
    available_space_width = available_space_right - available_space_left

    aadhaar_x = available_space_left + (available_space_width - aadhaar_text_width) // 2 - 5  
    aadhaar_y = int(h * 0.75)

    draw.text((x_right_of_photo, y_start), name_hindi, font=font_hindi_scaled, fill=(0, 0, 0))
    draw.text((x_right_of_photo, y_start + line_spacing), name_english, font=font_english_scaled, fill=(0, 0, 0))
    draw.text((x_right_of_photo, y_start + 2 * line_spacing), f"DOB: {dob}", font=font_dob_scaled, fill=(0, 0, 0))
    draw.text((x_right_of_photo, y_start + 3 * line_spacing), f"Gender: {gender}", font=font_english_scaled, fill=(0, 0, 0))
    draw.text((aadhaar_x, aadhaar_y), aadhaar_number, font=font_aadhaar_scaled, fill=(30, 30, 30))

  
    output_path = os.path.join(output_folder, f"aadhaar_card_{i}.png")
    card.save(output_path)
    print(f" Saved {output_path}")