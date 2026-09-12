# Golden Evaluation Set (200 Rows) Audit Report

**File Audited:** [`data/processed/golden_set_200.csv`](file:///c:/Users/hdnen/OneDrive/Desktop/hiver-sde-assignment/data/processed/golden_set_200.csv)  
**Audit Script:** [`src/audit_golden_set.py`](file:///c:/Users/hdnen/OneDrive/Desktop/hiver-sde-assignment/src/audit_golden_set.py)  
**Audit Date:** 2026-09-10  
**Integrity Guarantee:** **Read-only audit.** No labels or files were modified. Model suggestions are NOT treated as ground truth.

---

## 1. Executive Summary

An audit of the 200 human annotations in `golden_set_200.csv` was conducted to verify consistency with the intended sampling design and the established intent boundary rules.

### Key Metrics
- **Total Rows Evaluated:** 200
- **Agreements (Provisional Hypothesis == Reviewer Label):** **24 / 200 (12.0%)**
- **Disagreements (Provisional Hypothesis != Reviewer Label):** **176 / 200 (88.0%)**

---

## 2. Intended Sampling Design vs. Actual Reviewer Label Distribution

| Intent Label | Intended Count | Actual Count | Delta | Status | Description of Shift |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `other_general_inquiry` | **50** | **2** | **-48** | **SEVERE DEFICIT** | 48 out-of-scope tweets were classified into technical core intents |
| `app_crash_download_error` | **15** | **4** | **-11** | **DEFICIT** | 11 app crash/download rows were re-labeled into other classes |
| `billing_subscription_refund` | **15** | **10** | **-5** | DEFICIT | 5 billing rows re-labeled into other classes |
| `screen_touch_freeze` | **15** | **14** | **-1** | BALANCED | Near intended sample quota |
| `network_wifi_cellular` | **15** | **15** | **0** | **EXACT MATCH** | Exactly 15 rows assigned |
| `battery_drain_charging` | **15** | **17** | **+2** | BALANCED | Minor surplus |
| `keyboard_text_glitch` | **15** | **18** | **+3** | BALANCED | Minor surplus |
| `os_update_installation` | **15** | **25** | **+10** | SURPLUS | 10 additional rows absorbed from other categories |
| `apple_id_account_access` | **15** | **30** | **+15** | **LARGE SURPLUS** | Double the intended representation |
| `bluetooth_airpods_carplay` | **15** | **32** | **+17** | **LARGE SURPLUS** | Over double the intended representation |
| `hardware_buttons_audio` | **15** | **33** | **+18** | **LARGE SURPLUS** | Over double the intended representation |
| **TOTAL** | **200** | **200** | **0** | | |

---

## 3. Detailed Transition Analysis (Provisional $\rightarrow$ Reviewer Label)

The 176 disagreements group into the following top transition patterns:

| Rank | Provisional Intent | Reviewer Label | Count | Notable Characteristic |
| :-: | :--- | :--- | :---: | :--- |
| 1 | `other_general_inquiry` | `hardware_buttons_audio` | **13** | General queries/complaints assigned to hardware audio |
| 2 | `other_general_inquiry` | `apple_id_account_access` | **10** | General Mac/store queries assigned to Apple ID |
| 3 | `other_general_inquiry` | `bluetooth_airpods_carplay` | **9** | Unrelated inquiries assigned to Bluetooth |
| 4 | `other_general_inquiry` | `keyboard_text_glitch` | **7** | General update feedback assigned to keyboard glitch |
| 5 | `network_wifi_cellular` | `bluetooth_airpods_carplay` | **5** | Wi-Fi connectivity assigned to Bluetooth accessory |
| 6 | `battery_drain_charging` | `keyboard_text_glitch` | **4** | Charger/battery inquiries assigned to keyboard glitch |
| 7 | `network_wifi_cellular` | `hardware_buttons_audio` | **4** | Wi-Fi/data inquiries assigned to hardware buttons |
| 8 | `os_update_installation` | `hardware_buttons_audio` | **4** | Software update loops assigned to hardware buttons |
| 9 | `bluetooth_airpods_carplay` | `keyboard_text_glitch` | **4** | Bluetooth accessory tweets assigned to keyboard glitch |
| 10 | `hardware_buttons_audio` | `os_update_installation` | **4** | Physical button queries assigned to OS update |
| 11 | `billing_subscription_refund` | `apple_id_account_access` | **4** | Store billing/refund queries assigned to Apple ID |
| 12 | `other_general_inquiry` | `os_update_installation` | **4** | General complaints assigned to OS update |
| 13 | `other_general_inquiry` | `network_wifi_cellular` | **4** | Message sending / call drops assigned to network |
| 14 | `keyboard_text_glitch` | `battery_drain_charging` | **3** | Keyboard typing bugs assigned to battery drain |

---

## 4. Analysis: Consistency with Intended Sampling Design

**Conclusion:** The current annotations in `golden_set_200.csv` exhibit **severe systematic inconsistency** with the intended sampling design and established intent boundaries.

### Four Major Suspicious Patterns Identified:

#### Pattern 1: Collapse of the Out-of-Scope Class (`other_general_inquiry`)
- **Intended:** 50 out-of-scope examples representing pre-orders, shipping, store reservations, general praise/frustration, and non-technical chatter.
- **Observed:** 48 of the 50 examples were forced into technical core intents, leaving only 2 examples in `other_general_inquiry`.
- **Examples of Misclassification:**
  - **Row 194 (Tweet 2813642):** *"why was I charged for something I have a free trial on! That literally started today!"* $\rightarrow$ Labeled as `bluetooth_airpods_carplay`.
  - **Row 185 (Tweet 1847568):** *"J'en suis à ma 6ème tentative de ré-installation de macOS sur un MBP neuf..."* (macOS reinstallation) $\rightarrow$ Labeled as `bluetooth_airpods_carplay`.
  - **Row 186 (Tweet 2300973):** *"How do I download my preorders?? When I go to email and click download it acts like I need $$ on my Apple account..."* $\rightarrow$ Labeled as `apple_id_account_access`.
  - **Row 187 (Tweet 1357281):** *"movies I had on my iPod won't even go back on the iPod Stop making things harder and harder..."* $\rightarrow$ Labeled as `hardware_buttons_audio`.
  - **Row 196 (Tweet 2816437):** *"FaceID has stopped working on my iPhone X... I think my iPhone is defective."* $\rightarrow$ Labeled as `hardware_buttons_audio`.

#### Pattern 2: Severe Over-Representation in Three Classes
- `hardware_buttons_audio` (33 rows vs. 15 intended, **+18 surplus**)
- `bluetooth_airpods_carplay` (32 rows vs. 15 intended, **+17 surplus**)
- `apple_id_account_access` (30 rows vs. 15 intended, **+15 surplus**)
- Together, these 3 classes account for **95 out of 200 rows (47.5%)**, despite only representing 3.2% of the historical dataset distribution.

#### Pattern 3: Evidence of CLI Menu Offset / Rapid Keypress Misalignment
Examining the specific tweets and labels indicates that several consecutive rows received menu selections that did not match the query content:
- **Row 4 (Tweet 2206027):** *"Is there a fast/quick charger for #iPhone8Plus available separately? Could you send me an link plz?"* $\rightarrow$ Labeled as `keyboard_text_glitch`.
- **Row 11 (Tweet 1713769):** *"hey. How do I check my battery capacity ?"* $\rightarrow$ Labeled as `keyboard_text_glitch`.
- **Row 19 (Tweet 1543316):** *"why the FUCK is the letter i turning into a question mark"* (textbook iOS 11 'I' autocorrect bug) $\rightarrow$ Labeled as `network_wifi_cellular`.
- **Row 20 (Tweet 1683427):** *"Hey any chance you’re going fix this keyboard issue soon? I’m getting really sick of this shit."* $\rightarrow$ Labeled as `bluetooth_airpods_carplay`.
- **Row 21 (Tweet 1640746):** *"Sooo like when is going to fix this bs symbol-instead-of-a-letter!? Over it."* $\rightarrow$ Labeled as `apple_id_account_access`.
- **Row 22 (Tweet 1656547):** *"You guys really need to fix this, eye’m tired of typing eye all the damn time"* $\rightarrow$ Labeled as `billing_subscription_refund`.

#### Pattern 4: Accidental Input in `reviewer_action` Field
Several rows have `reviewer_action = 'n'`, such as Row 9, Row 13, and Row 18. This indicates that during CLI interaction, the user typed `'n'` (intended to answer the subsequent *"Escalation Expected? (y/n)"* prompt) into the *"Expected Action"* prompt instead.

---

## 5. Sample Disagreement Rows for Review

| Row | `tweet_id` | `clean_message` | `provisional_intent` | `reviewer_label` | Disagreement Diagnosis |
| :-: | :-: | :--- | :--- | :--- | :--- |
| **003** | `908482` | *"And it’s not just the battery. So many apps are not working well and the connectivity within apps is terrible."* | `battery_drain_charging` | `keyboard_text_glitch` | No keyboard mention; discusses battery, apps, and connectivity. |
| **004** | `2206027` | *"Is there a fast/quick charger for #iPhone8Plus available separately? Could you send me an link plz?"* | `battery_drain_charging` | `keyboard_text_glitch` | Explicitly asks for an iPhone 8 Plus fast charger. |
| **011** | `1713769` | *"hey. How do I check my battery capacity ?"* | `battery_drain_charging` | `keyboard_text_glitch` | Asks about battery health / capacity. |
| **019** | `1543316` | *"why the FUCK is the letter i turning into a question mark"* | `keyboard_text_glitch` | `network_wifi_cellular` | Core iOS 11 text replacement bug, not cellular/Wi-Fi. |
| **020** | `1683427` | *"Hey any chance you’re going fix this keyboard issue soon? I’m getting really sick of this shit."* | `keyboard_text_glitch` | `bluetooth_airpods_carplay` | Explicitly complains about keyboard, not Bluetooth/AirPods. |
| **021** | `1640746` | *"Sooo like when is going to fix this bs symbol-instead-of-a-letter!? Over it."* | `keyboard_text_glitch` | `apple_id_account_access` | Character substitution typing bug, not Apple ID login. |
| **022** | `1656547` | *"You guys really need to fix this, eye’m tired of typing eye all the damn time"* | `keyboard_text_glitch` | `billing_subscription_refund` | Typing workaround ('eye' instead of 'I'), not billing. |
| **185** | `1847568` | *"J'en suis à ma 6ème tentative de ré-installation de macOS sur un MBP neuf..."* | `other_general_inquiry` | `bluetooth_airpods_carplay` | macOS reinstallation issue in French, not Bluetooth. |
| **186** | `2300973` | *"How do I download my preorders?? When I go to email and click download it acts like I need $$ on my Apple account..."* | `other_general_inquiry` | `apple_id_account_access` | Preorder download / retail inquiry. |
| **194** | `2813642` | *"why was I charged for something I have a free trial on! That literally started today!"* | `other_general_inquiry` | `bluetooth_airpods_carplay` | Free trial billing dispute, not Bluetooth/AirPods. |

---

## 6. Recommendations for Human Re-Review

To ensure `golden_set_200.csv` serves as a gold standard evaluation benchmark:
1. **Do NOT run automated overwrites:** Keep human judgment as the ultimate authority.
2. **Review in a Spreadsheet (VS Code or Excel):** Rather than rapid CLI keypresses, opening `data/processed/golden_set_200.csv` in a tabular spreadsheet viewer allows full visual inspection of the message alongside the dropdown/cell values.
3. **Restore Out-of-Scope Rejections:** Verify that generic complaints, retail questions, preorders, and casual banter in rows 151–200 are appropriately assigned to `other_general_inquiry`.
4. **Re-align Shifted Rows:** Inspect rows 1–40 where menu indices appear to have shifted.
