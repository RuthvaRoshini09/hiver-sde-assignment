# Human Review & Golden Set Labeling Guide

This guide explains how to manually review and annotate the **200-row Golden Evaluation Set** in [`data/processed/golden_set_200.csv`](file:///c:/Users/hdnen/OneDrive/Desktop/hiver-sde-assignment/data/processed/golden_set_200.csv).

---

## 1. Allowed Intent Labels

Your manual annotation in `reviewer_label` must be one of these exact 11 labels:

| # | Allowed Intent Label | Category | Description |
| :-: | :--- | :--- | :--- |
| **1** | `battery_drain_charging` | Core | Rapid battery drain, device overheating, charging cable/port failure, phone dying fast |
| **2** | `keyboard_text_glitch` | Core | Autocorrect typing bug ('I' becomes 'A [?]'), predictive text glitch, keyboard lag |
| **3** | `network_wifi_cellular` | Core | Wi-Fi disconnects, cellular 'No Service', dropped calls, mobile data failure |
| **4** | `os_update_installation` | Core | iOS update failing to download/verify/install, device stuck on Apple logo boot loop |
| **5** | `bluetooth_airpods_carplay` | Core | AirPods audio dropping out, Bluetooth accessory pairing issues, CarPlay connection drops |
| **6** | `apple_id_account_access` | Core | Forgotten passwords, Apple ID locked or disabled, 2FA verification code missing |
| **7** | `hardware_buttons_audio` | Core | Physical buttons (Home button, volume, power) or internal mic/speaker/camera defect |
| **8** | `billing_subscription_refund` | Core | Unrecognized App Store charges, double billing, subscription cancellation, refund claims |
| **9** | `screen_touch_freeze` | Core | Unresponsive touch screen, frozen UI, black/blank display |
| **10** | `app_crash_download_error` | Core | Third-party apps crashing on open, App Store unable to download/update apps |
| **11** | `other_general_inquiry` | Out-of-Scope | Pre-order/shipping inquiries, general praise/complaints, store reservations, casual talk |

---

## 2. Intent Boundary Rules

When reviewing ambiguous messages, apply these 5 disambiguation rules:

1. **Financial Charge vs. Battery Charging:**
   - Mentions of *card, bank, account, money, dollar, billed, refund, purchase, subscription* $\rightarrow$ `billing_subscription_refund`.
   - Mentions of *battery, percentage, cable, charger, port, dying fast, overheat* $\rightarrow$ `battery_drain_charging`.

2. **Battery Drain Since Update vs. Update Installation:**
   - If the main complaint is **battery dying quickly or overheating** $\rightarrow$ `battery_drain_charging` (the update is temporal context).
   - If the complaint is that **the update failed to verify, threw an error, or froze the phone in a boot loop** $\rightarrow$ `os_update_installation`.

3. **Autocorrect / Keyboard Glitch vs. Frozen Screen:**
   - Text character substitution or autocorrect anomalies $\rightarrow$ `keyboard_text_glitch`.
   - Screen physically unresponsive to finger touch gestures $\rightarrow$ `screen_touch_freeze`.

4. **Internal Sound vs. Wireless Accessories:**
   - Internal speaker, receiver, earpiece, or microphone $\rightarrow$ `hardware_buttons_audio`.
   - AirPods, Bluetooth headphones, or vehicle CarPlay $\rightarrow$ `bluetooth_airpods_carplay`.

5. **Out-of-Scope & Transactional Inquiries:**
   - iPhone X preorders, delivery dates, store reservations, general praise/venting $\rightarrow$ `other_general_inquiry`.

---

## 3. Two Ways to Complete the Review

### Method A: Interactive CLI Labeling Tool (Recommended)
We built a dedicated interactive CLI tool that steps through each tweet, displays the text and provisional hypothesis, and prompts for your label, action, and escalation.

**To run the tool:**
```powershell
python src/labeling_tool.py
```
- Enter `1`–`11` to assign the intent.
- Press `Enter` to accept the recommended default action and escalation, or type a custom action.
- Press `s` to skip a row, `b` to go back, or `q` to save progress and exit.
- Your progress is automatically saved to `golden_set_200.csv` after every item. You can stop and resume anytime!

### Method B: Directly Editing the CSV File
You can directly open [`data/processed/golden_set_200.csv`](file:///c:/Users/hdnen/OneDrive/Desktop/hiver-sde-assignment/data/processed/golden_set_200.csv) in VS Code, Microsoft Excel, or Google Sheets:
1. For each row, fill in:
   - `reviewer_label`: One of the 11 allowed intent names.
   - `reviewer_action`: The expected support action (e.g. *Settings > Battery check*, *Reset Network Settings*, *reportaproblem.apple.com*).
   - `reviewer_escalation`: `True` or `False`.
   - `reviewer_notes`: (Optional) rationale or ambiguity notes.
2. Save the CSV file.

---

## 4. Validating and Finalizing the Dataset

Once you have labeled the rows, run the automated validator:
```powershell
python src/validate_golden_set.py
```

The validator will:
1. Verify that all 200 rows are annotated with valid intent labels.
2. Report the distribution of your human labels vs. the initial provisional hypotheses.
3. Automatically update `review_status` from `"PROVISIONAL - NOT HUMAN LABELLED"` to `"VERIFIED - GOLDEN"`.
4. Export the clean, verified benchmark dataset to [`data/processed/golden_set_verified.csv`](file:///c:/Users/hdnen/OneDrive/Desktop/hiver-sde-assignment/data/processed/golden_set_verified.csv).
