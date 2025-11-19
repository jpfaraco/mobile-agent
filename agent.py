import os
import time
import json
import base64
import re
import openai
import xml.etree.ElementTree as ET
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
from ppadb.client import Client as AdbClient
from fpdf import FPDF, XPos, YPos

# =================================================================================
# 1. VISION MODULE (Handles AI Interaction)
# =================================================================================

# Load environment variables from .env file
load_dotenv()
# Initialize the OpenAI client. It will automatically use the OPENAI_API_KEY
try:
    client = openai.OpenAI()
except openai.OpenAIError as e:
    print(f"❌ OpenAI API key not found or invalid. Please check your .env file or environment variables. Error: {e}")
    client = None

def encode_image_to_base64(image_path):
    """Encodes an image file to a base64 string for API calls."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        print(f"❌ Error encoding image {image_path}: {e}")
        return None

def analyze_screen(image_path, task_prompt, history, xml_content, actions_on_this_screen=[]):
    """
    Analyzes the screen using the screenshot, task, history, and the UI XML.
    This is the core "brain" of the agent.
    """
    if not client:
        print("❌ OpenAI client not initialized. Cannot analyze screen.")
        return None

    print("🤖 Sending screen and UI XML to AI for analysis...")
    
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return None
        
    history_string = "\n".join(f"- {h}" for h in history) if history else "None yet."

    visited_prompt_addition = ""
    if actions_on_this_screen:
        actions_string = "\n".join(f"- {a}" for a in actions_on_this_screen)
        visited_prompt_addition = f"""
        IMPORTANT: You have been on this screen before. Here are the actions you have taken from this screen in the past:
        {actions_string}

        These actions did not lead to progress. You are likely in a loop.
        You MUST choose a different action than the ones you have tried before.
        If there are no other reasonable actions to take, you MUST go back or finish the mission if you are truly stuck.
        """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are an expert mobile app automation agent. Your goal is to follow a user's mission step-by-step.

                    **Mission:** {task_prompt}

                    {visited_prompt_addition}

                    **Action History (What you have done so far):**
                    {history_string}

                    **Your Task:**
                    1.  **Reflect:** First, analyze the Mission, your Action History and where you currently are. In the `reflection` field, answer the question: "Based on my history and what I'm seeing now, have I completed all the steps required by the mission?"
                    2.  **Decide Status:** Based on your reflection, set the `status` field to "COMPLETE" or "IN_PROGRESS".
                    3.  **Think:** If the status is "IN_PROGRESS", explain your plan for the *next* action in the `thought` field.
                    4.  **Act:** Determine the single next action to take. Your available actions are "TAP", "GO_BACK", and "SCROLL". For a "TAP" action, you MUST find the correct node in the provided UI XML and return its literal 'bounds' attribute. Do not guess or make up coordinates. For "GO_BACK", you will navigate to the previous screen. For "SCROLL", you can scroll "down", "up", "left", or "right". If you don't see the element you are looking for, you might need to scroll.

                    **Output Format (JSON ONLY):**
                    {{
                      "history": {history_string},
                      "reflection": "A brief analysis of your progress, what you are looking at now vs. the mission requirements.",
                      "status": "IN_PROGRESS or COMPLETE",
                      "thought": "Your reasoning for the next specific action.",
                      "action": {{
                        "type": "TAP", "GO_BACK", or "SCROLL",
                        "bounds": "[x1,y1][x2,y2]", // Only for TAP
                        "direction": "down", "up", "left", or "right" // Only for SCROLL
                      }}
                    }}
                    """
                },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": f"Current screen's UI XML:\n```xml\n{xml_content}\n```" },
                        { "type": "image_url", "image_url": { "url": f"data:image/png;base64,{base64_image}" } }
                    ]
                }
            ],
            max_tokens=2000,
            response_format={ "type": "json_object" }
        )
        
        analysis = response.choices[0].message.content
        print("✅ AI analysis complete.")
        return analysis

    except Exception as e:
        print(f"❌ Error during AI analysis: {e}")
        return None

# =================================================================================
# 2. INTERACTION MODULE (Handles Device Communication)
# =================================================================================

def get_screen_size(device):
    """Gets the screen size of the device."""
    output = device.shell("wm size")
    match = re.search(r'Physical size: (\d+)x(\d+)', output)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def take_screenshot(device):
    """Takes a screenshot and saves it locally."""
    try:
        screenshot = device.screencap()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"screenshot_{timestamp}.png"
        screenshots_dir = "screenshots"
        if not os.path.exists(screenshots_dir):
            os.makedirs(screenshots_dir)
        file_path = os.path.join(screenshots_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(screenshot)
        print(f"✅ Screenshot saved to {file_path}")
        return file_path
    except Exception as e:
        print(f"❌ Failed to take screenshot: {e}")
        return None

def get_ui_xml(device):
    """Gets the device's UI XML dump for analysis."""
    try:
        device.shell("uiautomator dump /data/local/tmp/ui.xml")
        local_xml_path = "ui.xml"
        device.pull("/data/local/tmp/ui.xml", local_xml_path)
        with open(local_xml_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Failed to get UI XML: {e}")
        return None

def perform_tap(device, bounds_str, xml_content):
    """Performs a tap action based on the exact bounds string."""
    if not bounds_str:
        return "Failed to tap: bounds string was empty."
    try:
        coords = [int(n) for n in bounds_str.replace("][", ",").replace("[", "").replace("]", "").split(",")]
        x1, y1, x2, y2 = coords
        tap_x = (x1 + x2) // 2
        tap_y = (y1 + y2) // 2
        
        element_description = f"element with bounds {bounds_str}"
        try:
            root = ET.fromstring(xml_content)
            for node in root.iter('node'):
                bounds = node.attrib.get('bounds')
                if bounds == bounds_str:
                    text = node.attrib.get('text')
                    desc = node.attrib.get('content-desc')
                    if text:
                        element_description = f"element with text '{text}'"
                    elif desc:
                        element_description = f"element with description '{desc}'"
                    break
        except Exception:
            pass  # Ignore errors in finding element description

        print(f"👉 Executing precise tap at ({tap_x}, {tap_y})")
        device.shell(f"input tap {tap_x} {tap_y}")
        return f"Tapped on {element_description}"
    except Exception as e:
        error_message = f"Failed to tap due to invalid bounds: {bounds_str}. Error: {e}"
        print(f"❌ {error_message}")
        return error_message

def perform_go_back(device):
    """Performs a 'go back' action."""
    try:
        print("👉 Executing 'go back'")
        device.shell("input keyevent 4")
        return "Executed 'go back'"
    except Exception as e:
        error_message = f"Failed to execute 'go back'. Error: {e}"
        print(f"❌ {error_message}")
        return error_message

def perform_scroll(device, direction):
    """Performs a scroll action."""
    width, height = get_screen_size(device)
    if not width or not height:
        return "Failed to get screen size for scrolling."
    
    print(f"👉 Executing scroll '{direction}'")
    if direction == "down":
        start_x = width // 2
        start_y = int(height * 0.8)
        end_x = width // 2
        end_y = int(height * 0.2)
        duration = 500
        device.shell(f"input swipe {start_x} {start_y} {end_x} {end_y} {duration}")
        return "Scrolled down"
    elif direction == "up":
        start_x = width // 2
        start_y = int(height * 0.2)
        end_x = width // 2
        end_y = int(height * 0.8)
        duration = 500
        device.shell(f"input swipe {start_x} {start_y} {end_x} {end_y} {duration}")
        return "Scrolled up"
    else:
        return f"Scroll direction '{direction}' not supported."

# =================================================================================
# 3. REPORTING MODULE
# =================================================================================

def generate_pdf_report(mission, run_log, final_status):
    """Generates a PDF report of the agent's run."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    # Title
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, text="Agent Run Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)

    # Mission
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, text="Mission", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, text=mission)
    pdf.ln(10)

    # Steps
    for i, step in enumerate(run_log):
        if i > 0:
            pdf.add_page() # Add page break between steps

        pdf.set_font("Helvetica", 'B', 12)
        pdf.cell(0, 10, text=f"--- Step {step['step']} ---", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # Screenshot
        if os.path.exists(step['screenshot_path']):
            pdf.image(step['screenshot_path'], w=60) # Reduced width to 60
            pdf.ln(5)

        # Reflection
        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(0, 10, text="Reflection:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5, text=step['reflection'])
        pdf.ln(5)

        # Thought
        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(0, 10, text="Thought:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5, text=step['thought'])
        pdf.ln(5)

        # Action
        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(0, 10, text="Action:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5, text=step['action'])
        pdf.ln(10)

    # Final Status
    pdf.set_font("Helvetica", 'B', 14)
    pdf.cell(0, 10, text="Final Status", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, text=final_status, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Save the PDF
    report_dir = "reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"run_report_{timestamp}.pdf"
    file_path = os.path.join(report_dir, file_name)
    pdf.output(file_path)
    print(f"✅ Report saved to {file_path}")

# =================================================================================
# 4. AGENT CORE (Main Loop and Orchestration)
# =================================================================================

import argparse

def main():
    """Main function for the autonomous agent."""
    # --- Agent Configuration ---
    parser = argparse.ArgumentParser(description="Autonomous Mobile Agent")
    parser.add_argument("mission", type=str, help="The mission for the agent to perform.")
    parser.add_argument("--max-steps", type=int, default=15, help="Maximum number of steps for the agent to take.")
    args = parser.parse_args()
    mission = args.mission
    max_steps = args.max_steps

    # --- Initialize State ---
    action_history = []
    screen_history = {}
    run_log = []
    
    print(f"🚀 Starting agent with mission: {mission}")

    try:
        # --- Device Connection ---
        client = AdbClient(host="127.0.0.1", port=5037)
        devices = client.devices()
        if not devices:
            print("❌ No devices found. Please connect your device and enable USB debugging.")
            return
        device = devices[0]
        print(f"📱 Connected to device: {device.serial}")

        # --- Main Agent Loop ---
        final_status = "Unknown"
        for step in range(max_steps):
            print(f"\n--- Step {step + 1}/{max_steps} ---")
            
            # 1. Perceive (Screenshot + XML)
            screenshot_path = take_screenshot(device)
            xml_content = get_ui_xml(device)
            if not screenshot_path or not xml_content:
                print("❌ Perception failed. Cannot continue.")
                break
            
            # 2. Think (with history, screenshot, and XML)
            normalized_xml = re.sub(r'bounds="\[\d+,\d+\]\[\d+,\d+\]"', '', xml_content)
            normalized_xml = re.sub(r'\d', '', normalized_xml)
            
            actions_on_this_screen = screen_history.get(normalized_xml, [])
            analysis_string = analyze_screen(screenshot_path, mission, action_history, xml_content, actions_on_this_screen)

            if not analysis_string:
                print("❌ Thinking failed. Cannot get analysis from AI.")
                time.sleep(2)
                continue

            try:
                analysis_data = json.loads(analysis_string)
            except json.JSONDecodeError:
                print(f"❌ Error parsing JSON from AI: {analysis_string}")
                continue

            # Print AI's reasoning
            print(f"🤔 Reflection: {analysis_data.get('reflection')}")
            print(f"🧠 Thought: {analysis_data.get('thought')}")
            
            # 3. Act
            action = analysis_data.get("action", {})
            action_status = "No action taken."
            action_type = action.get("type")

            if analysis_data.get("status") == "IN_PROGRESS":
                if action_type == "TAP":
                    action_status = perform_tap(device, action.get("bounds"), xml_content)
                elif action_type == "GO_BACK":
                    action_status = perform_go_back(device)
                elif action_type == "SCROLL":
                    action_status = perform_scroll(device, action.get("direction", "down"))

            # 4. Update History and Log
            action_history.append(action_status)
            if normalized_xml not in screen_history:
                screen_history[normalized_xml] = []
            screen_history[normalized_xml].append(action_status)
            
            step_log = {
                "step": step + 1,
                "screenshot_path": screenshot_path,
                "reflection": analysis_data.get('reflection', 'N/A'),
                "thought": analysis_data.get('thought', 'N/A'),
                "action": action_status
            }
            run_log.append(step_log)

            # 5. Check for Mission Completion AFTER acting
            if analysis_data.get("status") == "COMPLETE":
                print("\n✅ Mission complete according to the agent's reflection.")
                final_status = "Successful"
                break
            
            time.sleep(2)  

        else:
            print(f"\n⚠️ Agent reached max steps ({max_steps}) without completing the mission.")
            final_status = "Failed (Max steps reached)"

    except Exception as e:
        print(f"\nAn unhandled error occurred during the agent's run: {e}")
        final_status = f"Failed (Error: {e})"
    
    finally:
        if run_log:
            generate_pdf_report(mission, run_log, final_status)


if __name__ == "__main__":
    main()
