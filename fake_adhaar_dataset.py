from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import random
import os


template_path = r"D:\empty1.png"    
output_folder = "tampered_aadhaar_cards"

male_photo_folder = r"males"
female_photo_folder = r"females"


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


fake_name_mapping = [
    ("Arjun Mehta", "अर्जुन मेहता", "Male"),
    ("Sneha Kapoor", "स्नेहा कपूर", "Female"),
    ("Karan Desai", "करण देसाई", "Male"),
    ("Riya Sharma", "रिया शर्मा", "Female"),
    ("Manish Jain", "मनीष जैन", "Male"),
    ("Priya Nair", "प्रिया नायर", "Female"),
    ("Suresh Rao", "सुरेश राव", "Male"),
    ("Divya Menon", "दिव्या मेनन", "Female"),
    ("Nikhil Bansal", "निखिल बंसल", "Male"),
    ("Aarti Chawla", "आरती चावला", "Female")
]


def random_aadhaar_number():
    """Generate random Aadhaar number."""
    return " ".join(["".join([str(random.randint(0, 9)) for _ in range(4)]) for _ in range(3)])

def random_dob():
    """Generate random date of birth."""
    return f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/{random.randint(1965, 2000)}"

def get_random_photo(gender):
    """Get random photo based on gender."""
    folder = male_photo_folder if gender == "Male" else female_photo_folder
    photo_list = os.listdir(folder)
    random_photo = random.choice(photo_list)
    photo = Image.open(os.path.join(folder, random_photo)).convert("RGB")
    photo = photo.resize((192, 210), Image.LANCZOS)
    return photo

def tamper_name(original_gender):
    """Replace original name with a random fake name (gender matched)."""
    filtered_fake_names = [f for f in fake_name_mapping if f[2] == original_gender]
    return random.choice(filtered_fake_names)

def tamper_gender(original_gender):
    """Flip gender."""
    return "Female" if original_gender == "Male" else "Male"

def tamper_logo(card):
    """Remove logo from Aadhaar card."""
    draw = ImageDraw.Draw(card)
    logo_area = (int(card.width * 0.05), int(card.height * 0.05), int(card.width * 0.25), int(card.height * 0.15))
    draw.rectangle(logo_area, fill=(255, 255, 255))
    return card

def tamper_slogan(card):
    """Remove slogan from Aadhaar card."""
    draw = ImageDraw.Draw(card)
    slogan_area = (int(card.width * 0.3), int(card.height * 0.9), int(card.width * 0.7), int(card.height * 0.95))
    draw.rectangle(slogan_area, fill=(255, 255, 255))
    return card

def tamper_qr(card):
    """Remove or distort QR code."""
    draw = ImageDraw.Draw(card)
    qr_area = (int(card.width * 0.7), int(card.height * 0.05), int(card.width * 0.95), int(card.height * 0.25))
    if random.choice([True, False]):
        # Blank out QR
        draw.rectangle(qr_area, fill=(255, 255, 255))
    else:
        # Add noise
        for _ in range(1000):
            x = random.randint(qr_area[0], qr_area[2])
            y = random.randint(qr_area[1], qr_area[3])
            draw.point((x, y), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    return card

def change_color_band(card):
    """Change color band to random color."""
    draw = ImageDraw.Draw(card)
    band_area = (0, int(card.height * 0.15), card.width, int(card.height * 0.22))
    random_color = tuple(random.choices(range(256), k=3))
    draw.rectangle(band_area, fill=random_color)
    return card

def apply_blur(card):
    """Apply blur effect."""
    return card.filter(ImageFilter.GaussianBlur(radius=2))

def apply_xerox_effect(card):
    """Apply xerox (black & white and blur) effect only to base card, not text."""
    base_card = card.convert("L")  # Convert to grayscale
    base_card = base_card.filter(ImageFilter.GaussianBlur(radius=1))
    return base_card.convert("RGB")  # Convert back to RGB for drawing text later

# Generate tampered Aadhaar cards
for i in range(1, 101):
    # Open template
    card = Image.open(template_path).copy()
    draw = ImageDraw.Draw(card)

    # Adjust font size relative to template size
    w, h = card.size
    scale = w / 600
    font_hindi_scaled = ImageFont.truetype(font_hindi_path, int(base_font_size * scale))
    font_english_scaled = ImageFont.truetype(font_english_path, int(base_font_size * scale))
    font_dob_scaled = ImageFont.truetype(font_dob_path, int((base_font_size - 3) * scale))
    font_aadhaar_scaled = ImageFont.truetype(font_aadhaar_path, int((base_font_size + 6) * scale))

    # Pick original details
    name_english, name_hindi, gender = random.choice(name_mapping)
    aadhaar_number = random_aadhaar_number()
    dob = random_dob()
    user_photo = get_random_photo(gender)

    # Insert photo
    photo_left, photo_top = int(w * 0.055), int(h * 0.27)
    card.paste(user_photo, (photo_left, photo_top))

    # Apply tampering based on batch
    batch = (i - 1) // 10 + 1

    if batch == 1:  # Name tampering
        fake_name_english, fake_name_hindi, _ = tamper_name(gender)
        name_english = fake_name_english
        name_hindi = fake_name_hindi
    elif batch == 2:  # Xerox simulation
        card = apply_xerox_effect(card)
    elif batch == 3:  # Blur effect
        card = apply_blur(card)
    elif batch == 4:  # Gender tampering
        gender = tamper_gender(gender)
    elif batch == 5:  # Aadhaar logo removal
        card = tamper_logo(card)
    elif batch == 6:  # Slogan removal
        card = tamper_slogan(card)
    elif batch == 7:  # Photo mismatch
        gender_alt = tamper_gender(gender)  # Force opposite gender photo
        user_photo = get_random_photo(gender_alt)
        card.paste(user_photo, (photo_left, photo_top))
    elif batch == 8:  # Aadhaar number mismatch
        aadhaar_number = random_aadhaar_number()
    elif batch == 9:  # DOB mismatch
        dob = random_dob()
    elif batch == 10:  # Name + DOB mismatch
        fake_name_english, fake_name_hindi, _ = tamper_name(gender)
        name_english = fake_name_english
        name_hindi = fake_name_hindi
        dob = random_dob()

    # Randomly tamper QR code for some cards
    if random.random() < 0.3:
        card = tamper_qr(card)

    # Randomly change color bands for 4-5 cards
    if random.random() < 0.05:
        card = change_color_band(card)

    # Text positions
    x_right_of_photo = photo_left + 192 + 30
    y_start = photo_top + 5
    line_spacing = int(h * 0.085)
    aadhaar_x = int(w * 0.35)
    aadhaar_y = int(h * 0.75)

    # Draw text (always draw after tampering)
    draw = ImageDraw.Draw(card)
    draw.text((x_right_of_photo, y_start), name_hindi, font=font_hindi_scaled, fill=(0, 0, 0))
    draw.text((x_right_of_photo, y_start + line_spacing), name_english, font=font_english_scaled, fill=(0, 0, 0))
    draw.text((x_right_of_photo, y_start + 2 * line_spacing), f"DOB: {dob}", font=font_dob_scaled, fill=(0, 0, 0))
    draw.text((x_right_of_photo, y_start + 3 * line_spacing), f"Gender: {gender}", font=font_english_scaled, fill=(0, 0, 0))
    draw.text((aadhaar_x, aadhaar_y), aadhaar_number, font=font_aadhaar_scaled, fill=(30, 30, 30))

    # Save card
    output_path = os.path.join(output_folder, f"tampered_aadhaar_card_{i}.png")
    card.save(output_path)
    print(f" Saved {output_path}")
