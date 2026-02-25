"""
Generate a synthetic Nigerian SMS dataset and train a Random Forest model.
This produces the .pkl files needed by the Fraudlock backend.

For production, replace this with your real nigerian_sms.csv dataset
and re-run random_forest.py from the machLearn repo.
"""
import os
import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── Synthetic Dataset ────────────────────────────────────────────────────────
# 1 = Spam/Phishing, 0 = Legitimate

SPAM_MESSAGES = [
    # BVN Scams
    "Your BVN has been flagged for suspicious activity. Send your BVN and ATM PIN to 08012345678 to resolve.",
    "CBN Alert: Your BVN is about to be blocked. Dial *565*0# or send BVN to 09087654321 now.",
    "Dear customer, verify your BVN immediately at cbn-verify.com to avoid account suspension.",
    "URGENT: Your BVN has been linked to fraud. Call 07034567890 to clear your name before arrest.",
    "Your bank account will be frozen in 24hrs if you dont verify your BVN. Send to 08098765432.",
    "CBN: All accounts without BVN update will be closed. Send your BVN, Date of Birth, and PIN to update.",
    "Your BVN needs urgent verification. Click https://update-bvn-ng.com to verify now before your account is blocked.",
    "Final Notice: Your BVN has expired. Revalidate now at https://cbn-bvn-update.ng or lose access to your account.",

    # Bank Phishing
    "GTBank: Your account has been suspended. Click https://gtb-verify.net to restore access immediately.",
    "First Bank Alert: Unauthorized login detected. Verify at https://firstbank-secure.com or account will be locked.",
    "Access Bank: Your transfer of N500,000 is pending. Confirm with OTP sent to 08012345678.",
    "UBA Notice: Your account has been compromised. Reset your password at https://uba-reset.com now.",
    "Zenith Bank: You have N2,000,000 pending. Send your account details to release funds.",
    "Dear customer, your Zenith Bank token has expired. Visit https://zenith-token.com to renew immediately.",
    "Kuda Bank security alert: Someone tried to access your account. Verify identity at https://kuda-alert.ng.",
    "Fidelity Bank: Your OTP is 847362. If you did not request this, call 08033333333 immediately.",
    "Opay Alert: Your account balance of N350,000 requires verification. Send ATM PIN to confirm.",
    "Sterling Bank: Transaction of N1,500,000 flagged. Confirm or cancel at https://sterling-alert.ng.",

    # JAMB Scams
    "JAMB: Your UTME result has been upgraded to 280. Pay N3,500 to 08012345678 to claim.",
    "Congratulations! JAMB has awarded you admission. Pay N5,000 acceptance fee to 0901234567.",
    "JAMB Regularization: Pay N15,000 to regularize your result. Transfer to account 0123456789 GTB.",
    "Your JAMB result has been withheld. Send N10,000 to clear it. Call Prof Oloyede at 08098765432.",
    "JAMB 2024: You scored 310! Send N8,000 for certificate collection to 08055667788.",

    # Prize / Lottery
    "Congratulations! You won N5,000,000 in the MTN Mega Promo! Send your details to claim your prize now.",
    "GLO WINNER! You just won N2,000,000 airtime. Forward this to 10 people and call 07011223344.",
    "You are the lucky winner of N10,000,000 from 9Mobile. Send name, account number, and BVN to claim.",
    "MTN: Your number won N500,000! To claim, recharge N5,000 worth of credit and call 08099887766.",
    "Airtel Jackpot! You won a brand new car. Pay N25,000 delivery fee to 0812345678 to receive.",
    "Dear customer, you have been selected for a cash prize of N1,000,000. Reply YES to claim.",
    "DSTV promo: You won a free 12-month subscription! Send your smartcard number and N2,000 processing fee.",

    # Ponzi / Investment
    "Invest N10,000 and earn N100,000 in 24 hours! 100% guaranteed. Join now at https://quickcash-ng.com.",
    "Double your money in 2 hours! Send N50,000 to account 0123456789 and receive N100,000 back guaranteed.",
    "MBA Forex: Invest N20,000 and earn N200,000 weekly. No risk! Call 08033445566 to start.",
    "Crowdfunding opportunity! Invest just N5,000 and get N50,000 in 7 days. Send to 0198765432 Zenith Bank.",
    "Chipper Cash promo: Deposit N30,000 and get N90,000 bonus. Limited slots available! Act now.",

    # OTP / PIN Theft
    "Your OTP is 482917. If you did not request this, call 08012345678 to cancel the transaction.",
    "Please share the OTP sent to your phone to complete your pending transfer of N250,000.",
    "Your ATM PIN has been compromised. Send current PIN to 08099988877 to get a new one.",
    "We detected unusual activity. Reply with your 4-digit PIN to secure your account.",
    "Transaction alert: N150,000 debit. If this wasn't you, send your PIN and OTP to 0803344556.",

    # Fake Delivery
    "Your DHL package has arrived at Lagos. Pay N3,500 customs fee to 08012345678 for delivery.",
    "GIG Logistics: Your parcel requires N5,000 clearance. Send to 12345678 GTBank to release.",
    "Amazon delivery for you in Nigeria! Pay N8,000 shipping to account 0987654321 UBA.",
    "Jumia: Your order has been shipped. Confirm address by clicking https://jumia-tracking.ng/verify.",
    "FedEx: Package held at customs. Pay N12,000 duty to release. Call 09011223344.",

    # Government Impersonation
    "EFCC: You are under investigation. Pay N50,000 fine to avoid arrest. Account 0123456789 First Bank.",
    "NCS Customs: Your seized goods can be released. Pay N100,000 to account 5678901234 Zenith.",
    "Federal Government Grant: You qualify for N500,000 grant. Send processing fee of N10,000.",
    "FRSC: Your driver's license renewal is overdue. Pay N15,000 fine at https://frsc-renewal.ng.",
    "Immigration: Your passport application requires N20,000 expedite fee. Transfer to 0112233445 Access.",

    # Pidgin scams
    "Oga, your account don block! Send your BVN quick quick make we fix am for you.",
    "See money wey dey wait you! Send N5,000 come collect N500,000. No dulling!",
    "Your number don win N2,000,000! Call 08012345678 now now to claim am before e expire.",
    "Abeg verify your account make dem no close am. Send your PIN to 09087654321.",
    "Glo bonus: Send your recharge card PIN to 07034567890 get double bonus immediately.",

    # Yoruba scams
    "E ku ile o! Owo N2,000,000 ti de fun yin. Fi BVN yin ranse lati gba.",
    "Ikilọ: Account yin ti di. Fi PIN yin ranse si 08012345678 ki a le tu silẹ.",

    # Hausa scams
    "Ka samu N1,000,000! Aika BVN dinka yanzu don karbar kudin.",
    "Asusun ka na cikin hatsari. Aika lambar PIN zuwa 08012345678 don tabbatarwa.",

    # Igbo scams
    "Nwanne, ego N3,000,000 dị n'akpa gị. Zitere anyị BVN gị ugbu a.",
    "Akaụntụ gị nwere nsogbu. Zitere PIN gị na 09087654321 ka anyị nyere gị aka.",

    # Loan harassment
    "You owe N15,000 from EaseMoni. Pay now or we expose you to all your contacts!",
    "URGENT: FairMoney loan overdue. Your photo and contacts will be shared if not paid in 1 hour.",
    "PalmCredit: Your N10,000 loan is 3 days overdue. Your employer has been notified. Pay now!",
    "Branch loan default: We are sending recovery agents to your home address. Pay N8,000 now.",

    # General phishing
    "Click here to claim your free iPhone 15: https://free-iphone-ng.com. Limited time offer!",
    "You have been selected for a N200,000 scholarship. Apply now at https://scholarship-ng.com/apply.",
    "WhatsApp is shutting down your account. Forward this to 20 contacts to keep it active.",
    "Facebook security: Your account will be deleted in 24hrs. Verify at https://fb-verify-ng.com.",
    "Instagram alert: Someone is trying to hack your account. Login at https://ig-secure.com to protect it.",
    "Verify now or your MTN line will be disconnected within 48 hours. Call 08055443322 immediately.",
    "Your electricity meter is about to be disconnected. Pay N25,000 prepaid at https://nepa-pay.ng.",
    "Confirm your NIN number at https://nin-update.ng or your SIM will be blocked by NCC.",
]

LEGIT_MESSAGES = [
    # Personal conversations
    "Hi Emeka, are we still on for the meeting tomorrow at 3pm? Let me know if the time works.",
    "Good morning Mama. Just checking on you. How is everyone at home?",
    "Bros, I don land Lagos. Which bus I go take from Ojuelegba to Ikeja?",
    "Happy birthday Ada! Wishing you a wonderful year ahead. God bless you.",
    "Aunty, please can you help me pick the children from school today? I'm stuck in traffic.",
    "Hey fam, the burial is confirmed for Saturday. Wear white and black. See you there.",
    "Bro please send me that document we discussed yesterday. I need it before the presentation.",
    "Traffic is terrible today on Third Mainland. I'll be late for the meeting, maybe 30 mins.",
    "Just saw your missed call. What's up? I was in a lecture.",
    "Good evening sir. I wanted to ask about the project deadline. Is it still Friday?",
    "How far Tunde? You dey come the hangout tonight? Everybody dey wait you o.",
    "Please remind me to buy garri and oil when we go to market tomorrow.",
    "Congrats on your new job! Very proud of you. Let's celebrate this weekend.",
    "I just landed in Abuja. The weather is cold here. Will call you when I settle in.",
    "Doc said the test results are fine. Nothing to worry about. Thank God!",
    "Can you please send me the gate code? I'm outside your estate.",
    "The landlord said rent is due by month end. Have you saved up your share?",
    "Let me know when you get to the office. I'll come meet you there.",

    # Bank alerts (legitimate)
    "GTBank: You have received N45,000 from Adebayo Olamide. Ref: TRF/2024/123456.",
    "Your account ***4521 has been credited with N120,000.00. Available balance: N350,000.00.",
    "Access Bank Alert: Debit of N5,000 from account ***7890. Balance: N28,500.",
    "Kuda: Transfer of N15,000 to Chinedu Okafor successful. Reference: KDA20241234.",
    "UBA Alert: N80,000.00 deposit confirmed. Available balance: N250,000.00. Thank you for banking with us.",
    "OPay: Your wallet was funded with N10,000.00. Balance: N23,500.00.",
    "First Bank Alert: N3,500 POS purchase at Shoprite Ikeja. Balance: N65,800.00. Info: 08039001234.",
    "Zenith Bank: Salary credit of N185,000.00 from ABC Company Ltd. Ref: SAL/2024/567.",

    # Service notifications
    "Your MTN data bundle of 1.5GB has been activated. Valid for 30 days. Dial *131# to check balance.",
    "Airtel: Your N1,000 recharge was successful. Airtime balance: N1,250.",
    "GLO: You have 500MB bonus data valid for 7 days. Enjoy!",
    "9Mobile reminder: Your monthly plan of N3,500 renews tomorrow. Ensure sufficient balance.",
    "DSTV: Your subscription has been renewed for 1 month. Package: Compact. Enjoy your viewing.",
    "Your WAEC result is ready. Check at www.waecdirect.org with your exam number.",
    "Lagos State Transport: Your vehicle papers expire on 30/06/2024. Renew at any MVAA office.",
    "PHCN: Your prepaid meter token is 1234-5678-9012-3456. Units: 50kWh. Thank you.",

    # Delivery confirmations
    "Your GIG Logistics package has been shipped. Track at giglogistics.com. Waybill: GIG123456789.",
    "Jumia: Your order #JUM-987654 is out for delivery. Expected arrival: Today before 6pm.",

    # Work / professional
    "Meeting rescheduled to 2pm tomorrow. Updated agenda has been shared on the group.",
    "Your leave request for March 15-20 has been approved. Please ensure proper handover.",
    "Payslip for February 2024 is available on the HR portal. Salary credited to your account.",
    "Reminder: Staff training on cybersecurity this Wednesday at 10am. Conference Room B.",
    "Your application for the position has been received. We will contact you within 2 weeks.",

    # Church / community
    "House fellowship holds at Bro Emeka's house this Friday by 6pm. Bring your Bible.",
    "The community association meeting is Saturday 4pm at the town hall. Attendance is mandatory.",
    "Church building fund: We appreciate your generous donation of N50,000. God bless you.",

    # Weather and info
    "NiMet: Heavy rainfall expected in Lagos and Ogun states today. Stay safe.",
    "Traffic update: Lekki-Epe expressway is clear. Estimated travel time to VI: 35 mins.",
    "INEC: Voter registration closes on January 31. Visit the nearest INEC office to register.",

    # Pidgin legit
    "How far? I dey come your side now. Wait for me for gate.",
    "Oya send me the address make I put am for Google Maps.",
    "E don reach your turn to buy drink for the boys. No run o!",
    "Mama say make you buy rice come house. She dey cook jollof today.",
    "I just see the news. Na wa o. Make we pray for Nigeria.",

    # Yoruba legit
    "E kaale Ma. Se daadaa ni? Mo fe beere nipa ipade ola.",
    "Bawo ni ise? Mo ti gba owo naa. E se pupo.",

    # Hausa legit
    "Sannu da zuwa! Mun gode da zuwan ka. Allah ya kara mana arziki.",
    "Yaya gida? Ina so in tambaye ka game da aikin da muka tattauna.",

    # Igbo legit
    "Kedu ka ị mere? Achọrọ m ịjụ maka nzukọ ụbọchị Satọde.",
    "Daalụ nwanne m. Ego a abatala. Chukwu gozie gị.",
]

def build_dataset():
    """Build pandas DataFrame from the synthetic messages."""
    messages = []
    labels = []
    for msg in SPAM_MESSAGES:
        messages.append(msg)
        labels.append(1)
    for msg in LEGIT_MESSAGES:
        messages.append(msg)
        labels.append(0)
    return pd.DataFrame({"Message": messages, "Spam": labels})


def preprocess_text(text):
    """Clean and normalize text — same as backend/views.py."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\b\d{10,11}\b', ' PHONE ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'[$₦£€]\d+|\d+\s*(naira|dollars|pounds)', ' MONEY ', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', ' NUMBER ', text)
    return text


def main():
    print("=" * 60)
    print("Fraudlock — Random Forest SMS Model Training")
    print("=" * 60)

    # Build dataset
    print("\n[1/5] Building dataset...")
    data = build_dataset()
    print(f"  ✓ {len(data)} messages ({len(SPAM_MESSAGES)} spam, {len(LEGIT_MESSAGES)} legit)")

    # Preprocess
    print("\n[2/5] Preprocessing messages...")
    data['Message_processed'] = data['Message'].apply(preprocess_text)

    # Split
    print("\n[3/5] Splitting data...")
    X = data['Message_processed']
    y = data['Spam']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing:  {len(X_test)} samples")

    # TF-IDF
    print("\n[4/5] Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"  Feature count: {X_train_vec.shape[1]}")

    # Train
    print("\n[5/5] Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    rf_model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'=' * 60}")
    print(f"ACCURACY: {accuracy:.2%}")
    print(f"{'=' * 60}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Spam"]))

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "ml_api", "ml")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "sms_phishing_model_rf.pkl")
    vec_path   = os.path.join(output_dir, "tfidf_vectorizer_rf.pkl")

    joblib.dump(rf_model, model_path)
    joblib.dump(vectorizer, vec_path)

    print(f"\n✓ Model saved:      {model_path}")
    print(f"✓ Vectorizer saved: {vec_path}")
    print("\nDone! Restart the Django server to load the new model.")


if __name__ == "__main__":
    main()
