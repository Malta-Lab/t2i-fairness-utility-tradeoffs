import streamlit as st
import os
import json
import tempfile
import shutil
import zipfile
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv
from main import (
    ImageEvaluator,
    get_available_metrics,
    create_pareto_chart_data,
    get_gpu_info,
)

load_dotenv()

COLOR_OPTIONS = [
    ("🟠 Orange", "#FF8C00"),
    ("🔴 Red", "#FF4444"),
    ("🟢 Green", "#28A745"),
    ("🟣 Purple", "#8B5CF6"),
    ("🟡 Yellow", "#FFC107"),
    ("🔵 Light Blue", "#17A2B8"),
    ("🟤 Brown", "#8B4513"),
    ("🟣 Pink", "#FF69B4"),
    ("⚫ Dark Gray", "#6C757D"),
]

SYMBOL_OPTIONS = [
    ("● Circle", "circle"),
    ("■ Square", "square"),
    ("▲ Triangle Up", "triangle-up"),
    ("▼ Triangle Down", "triangle-down"),
    ("♦ Diamond", "diamond"),
    ("✚ Cross", "cross"),
    ("✖ X", "x"),
    ("★ Star", "star"),
    ("◆ Diamond Open", "diamond-open"),
    ("☰ Hexagon", "hexagon"),
]


st.set_page_config(
    page_title="Benchmarking the trade-off between selected metrics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title(
    "🎯 Framework for Benchmarking Fairness-Utility Trade-offs in Text-to-Image Models via Pareto Frontiers"
)
st.markdown(
    """
This application evaluates AI-generated images for utility and fairness metrics.
Upload a folder containing images following the naming convention: `topic_parameters_seedX.png`

**Example**: `firefighter_cfg3.0_sld1000.0_seed1.png`
"""
)

try:
    gpu_info = get_gpu_info()
except Exception as e:
    st.error(f"Error getting GPU information: {e}")
    gpu_info = []


st.sidebar.header("Configuration")

st.sidebar.subheader("🖥️ GPU Selection")

if st.sidebar.button("🔄 Refresh GPU Status"):
    gpu_info = get_gpu_info()
    st.rerun()

if gpu_info:
    st.sidebar.success(f"✅ Found {len(gpu_info)} GPU(s)")

    gpu_options = ["CPU Only"]
    gpu_details = []

    for gpu in gpu_info:
        option_text = f"GPU {gpu['index']}: {gpu['name']}"
        gpu_options.append(option_text)

        gpu_detail = {
            "GPU": f"GPU {gpu['index']}",
            "Name": gpu["name"],
            "Memory Usage": f"{gpu['memory_used']:.0f} / {gpu['memory_total']:.0f} MB",
            "Memory Free": f"{gpu['memory_free']:.0f} MB",
            "GPU Utilization": f"{gpu['utilization']}%",
            "Memory Usage %": f"{gpu['memory_usage_percent']:.1f}%",
        }
        gpu_details.append(gpu_detail)

    if gpu_details:
        st.sidebar.markdown("**GPU Status:**")
        gpu_df = pd.DataFrame(gpu_details)
        st.sidebar.dataframe(gpu_df, use_container_width=True)

    selected_gpu_option = st.sidebar.selectbox(
        "Select GPU to use:",
        gpu_options,
        help="Choose which GPU to use for processing. CPU Only will use CPU for all computations.",
    )

    if selected_gpu_option == "CPU Only":
        selected_gpu_index = None
        st.sidebar.info("💡 Using CPU for processing")
    else:
        selected_gpu_index = int(selected_gpu_option.split(":")[0].split()[-1])
        selected_gpu = gpu_info[selected_gpu_index]

        st.sidebar.info(
            f"""
        **Selected GPU {selected_gpu_index}:**
        - {selected_gpu['name']}
        - Free Memory: {selected_gpu['memory_free']:.0f} MB
        - Current Usage: {selected_gpu['utilization']}%
        """
        )

        if selected_gpu["utilization"] > 80:
            st.sidebar.warning(
                "⚠️ Selected GPU has high utilization. Consider using a different GPU."
            )

        if selected_gpu["memory_usage_percent"] > 90:
            st.sidebar.warning(
                "⚠️ Selected GPU has low free memory. Processing might be slow."
            )
else:
    st.sidebar.warning("⚠️ No NVIDIA GPUs detected. Using CPU only.")
    selected_gpu_index = None

st.sidebar.markdown("---")

api_provider = st.sidebar.selectbox(
    "API Provider",
    ["OpenAI", "OpenRouter", "Google Gemini"],
    help="Select the API provider for demographic analysis",
)


def get_env_api_key(provider):
    """Get API key from environment variables based on provider"""
    if provider == "OpenAI":
        return os.getenv("OPENAI_API_KEY", "")
    elif provider == "OpenRouter":
        return os.getenv("OPENROUTER_API_KEY", "")
    elif provider == "Google Gemini":
        return os.getenv("GEMINI_API_KEY", "")
    return ""


env_api_key = get_env_api_key(api_provider)

if env_api_key:
    st.sidebar.success(f"✅ API key auto-loaded from .env for {api_provider}")
    masked_key = (
        f"{env_api_key[:20]}...{env_api_key[-4:]}"
        if len(env_api_key) > 24
        else env_api_key[:10] + "..."
    )
    st.sidebar.info(f"🔑 Key: {masked_key}")

api_key = st.sidebar.text_input(
    "API Key",
    value=env_api_key,
    type="password",
    help=f"API key for {api_provider}. Auto-loaded from .env if available, or enter manually.",
)

if api_provider == "OpenAI":
    model_name = st.sidebar.selectbox(
        "Model",
        ["gpt-4o-2024-05-13", "gpt-4o", "gpt-4-vision-preview", "gpt-4o-mini"],
        help="Select the OpenAI model",
    )
elif api_provider == "OpenRouter":
    model_name = st.sidebar.text_input(
        "Model Name",
        value="anthropic/claude-3-sonnet",
        help="Enter the model name from OpenRouter",
    )
else:
    model_name = st.sidebar.selectbox(
        "Model",
        ["gemini-2.0-flash", "gemini-2.5-pro-preview-06-05"],
        help="Select the Gemini model",
    )

st.sidebar.markdown("---")

if hasattr(st.session_state, "results") and hasattr(
    st.session_state, "all_available_metrics"
):
    st.sidebar.header("📊 Metrics to Display")
    st.sidebar.info("Select which metrics to show in tables and charts")

    available_metrics = st.session_state.all_available_metrics
    display_metrics = []

    metric_display_names = {
        "clip_score": "🎯 CLIP Score",
        "entropy_fairness": "⚖️ Normalized Entropy",
        "kl_fairness": "📊 KL Divergence",
    }

    for metric in available_metrics:
        display_name = metric_display_names.get(metric, metric)
        if st.sidebar.checkbox(
            display_name,
            value=True,
            key=f"display_{metric}",
            help=f"Show {display_name} in results",
        ):
            display_metrics.append(metric)

    st.session_state.display_metrics = display_metrics

    if len(display_metrics) == 2:
        st.sidebar.success("✅ Pareto chart available (2 metrics selected)")
    elif len(display_metrics) < 2:
        st.sidebar.info("💡 Select 2 metrics to see Pareto chart")
    else:
        st.sidebar.warning("⚠️ Pareto chart needs exactly 2 metrics")

st.sidebar.markdown("---")

st.header("📊 Input Method")
input_method = st.radio(
    "Choose how to provide data:",
    ["🖼️ Process New Images", "📄 Upload Previous Results (JSON)"],
    horizontal=True,
    help="Process new images to generate results, or upload a previously saved JSON file to visualize existing results",
)

if input_method == "🖼️ Process New Images":
    st.sidebar.header("Metrics Selection")
    available_metrics = get_available_metrics()

    if not available_metrics:
        st.sidebar.error(
            "No metrics available. Please install required dependencies (torch, clip, openai)."
        )
        st.stop()

    selected_metrics = []
    for metric in available_metrics:
        if st.sidebar.checkbox(
            metric["name"], help=metric["description"], key=metric["key"]
        ):
            selected_metrics.append(metric["key"])

    if not selected_metrics:
        st.sidebar.warning("Please select at least one metric to evaluate.")

    st.header("📁 Upload Images")

    st.info(
        "💡 **File Requirements:**\n"
        "- Supported formats: PNG, JPG, JPEG, BMP, TIFF\n"
        "- Naming convention: `topic_parameters_seedX.extension`\n"
        "- Example: `nurse_g3.0_sld0.0_seed42.jpg`\n"
        "- **ZIP Support**: Can contain multiple folders with images"
    )

    st.warning(
        "⚠️ **Upload Limit**: 200MB per file (Streamlit default). For larger files, see sidebar for configuration instructions."
    )

    upload_method = st.radio(
        "Upload Method",
        ["Select Individual Files", "Upload Folder as ZIP"],
        horizontal=True,
        help="ZIP method supports complex folder structures with multiple subdirectories",
    )

    temp_folder = None
    uploaded_files = None

    if upload_method == "Upload Folder as ZIP":
        uploaded_zip = st.file_uploader(
            "Upload a ZIP file containing your images",
            type=["zip"],
            help="ZIP can contain multiple folders. All images in all subdirectories will be processed automatically.",
        )

        if uploaded_zip:
            temp_folder = tempfile.mkdtemp()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                tmp_zip.write(uploaded_zip.read())
                tmp_zip_path = tmp_zip.name

            try:
                with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
                    zip_ref.extractall(temp_folder)

                total_image_files = []
                for root, dirs, files in os.walk(temp_folder):
                    image_files_in_dir = [
                        f
                        for f in files
                        if f.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
                        )
                    ]
                    for img_file in image_files_in_dir:
                        total_image_files.append(os.path.join(root, img_file))

                if total_image_files:
                    st.success(
                        f"✅ ZIP file extracted successfully! Found {len(total_image_files)} images across all subdirectories."
                    )

                    dir_structure = {}
                    for img_path in total_image_files:
                        rel_path = os.path.relpath(img_path, temp_folder)
                        dir_name = os.path.dirname(rel_path)
                        if dir_name == "":
                            dir_name = "(root)"
                        if dir_name not in dir_structure:
                            dir_structure[dir_name] = 0
                        dir_structure[dir_name] += 1

                    if len(dir_structure) > 1:
                        st.info(
                            "📁 **Directory Structure:**\n"
                            + "\n".join(
                                [
                                    f"- `{folder}`: {count} images"
                                    for folder, count in sorted(dir_structure.items())
                                ]
                            )
                        )
                else:
                    st.error("❌ No image files found in the ZIP file.")
                    temp_folder = None

            except Exception as e:
                st.error(f"Error extracting ZIP file: {e}")
                temp_folder = None
            finally:
                os.unlink(tmp_zip_path)

    else:
        uploaded_files = st.file_uploader(
            "Upload image files",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Select multiple image files following the naming convention",
        )

        if uploaded_files:
            try:

                temp_folder = tempfile.mkdtemp()

                successful_uploads = 0
                failed_uploads = []

                for uploaded_file in uploaded_files:
                    try:
                        safe_filename = uploaded_file.name.replace(" ", "_")
                        file_path = os.path.join(temp_folder, safe_filename)

                        file_content = uploaded_file.read()

                        uploaded_file.seek(0)

                        with open(file_path, "wb") as f:
                            f.write(file_content)

                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            successful_uploads += 1
                        else:
                            failed_uploads.append(
                                f"{uploaded_file.name} (file not written properly)"
                            )

                    except Exception as e:
                        failed_uploads.append(f"{uploaded_file.name} ({str(e)})")

                if successful_uploads > 0:
                    st.success(f"✅ Successfully uploaded {successful_uploads} files!")
                    if failed_uploads:
                        st.warning(
                            f"⚠️ Failed to upload {len(failed_uploads)} files: {', '.join(failed_uploads)}"
                        )
                else:
                    st.error("❌ Failed to upload any files. Please try again.")
                    temp_folder = None

            except Exception as e:
                st.error(f"Error during file upload: {e}")
                temp_folder = None

elif input_method == "📄 Upload Previous Results (JSON)":
    selected_metrics = getattr(st.session_state, "selected_metrics", [])

    st.header("📄 Upload Results JSON")

    st.info(
        "**JSON Requirements:**\n"
        "- Upload a previously generated evaluation results file\n"
        "- Must follow the standard format with 'topic', 'configurations', and 'summary' sections\n"
        "- Generated from this tool's 'Download Results (JSON)' option"
    )

    st.subheader("Evaluated Configurations File (Blue Points)")
    uploaded_json = st.file_uploader(
        "Upload main results JSON file",
        type=["json"],
        help="Upload the main results file to visualize and analyze the data",
        key="main_json",
    )

    st.subheader("Comparison Files (Colored Points)")
    st.info(
        "💡 Add additional files to compare against your main results. Each file will appear with your chosen color on the chart."
    )

    if "comparison_files" not in st.session_state:
        st.session_state.comparison_files = []

    num_comparison_files = st.number_input(
        "Number of comparison files",
        min_value=0,
        max_value=5,
        value=len(st.session_state.comparison_files),
        help="Select how many comparison files you want to add",
    )

    while len(st.session_state.comparison_files) < num_comparison_files:
        default_color_index = len(st.session_state.comparison_files) % len(
            COLOR_OPTIONS
        )
        default_color = COLOR_OPTIONS[default_color_index][1]
        default_symbol_index = len(st.session_state.comparison_files) % len(
            SYMBOL_OPTIONS
        )
        default_symbol = SYMBOL_OPTIONS[default_symbol_index][1]
        st.session_state.comparison_files.append(
            {
                "file": None,
                "name": "",
                "data": None,
                "color": default_color,
                "symbol": default_symbol,
            }
        )
    while len(st.session_state.comparison_files) > num_comparison_files:
        st.session_state.comparison_files.pop()

    for i in range(num_comparison_files):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1.2])

        with col1:
            comparison_file = st.file_uploader(
                f"Comparison file {i+1}",
                type=["json"],
                help="Upload a JSON file to compare",
                key=f"comparison_json_{i}",
            )
            st.session_state.comparison_files[i]["file"] = comparison_file

        with col2:
            if "color" not in st.session_state.comparison_files[i]:
                default_color_index = i % len(COLOR_OPTIONS)
                st.session_state.comparison_files[i]["color"] = COLOR_OPTIONS[
                    default_color_index
                ][1]

            current_color = st.session_state.comparison_files[i]["color"]
            current_index = 0
            for idx, (name, color) in enumerate(COLOR_OPTIONS):
                if color == current_color:
                    current_index = idx
                    break

            selected_color_index = st.selectbox(
                f"Color {i+1}",
                range(len(COLOR_OPTIONS)),
                format_func=lambda x: COLOR_OPTIONS[x][0],
                index=current_index,
                help="Choose color for this comparison point",
                key=f"comparison_color_{i}",
            )
            st.session_state.comparison_files[i]["color"] = COLOR_OPTIONS[
                selected_color_index
            ][1]

        with col3:
            if "symbol" not in st.session_state.comparison_files[i]:
                default_symbol_index = i % len(SYMBOL_OPTIONS)
                st.session_state.comparison_files[i]["symbol"] = SYMBOL_OPTIONS[
                    default_symbol_index
                ][1]

            current_symbol = st.session_state.comparison_files[i]["symbol"]
            current_symbol_index = 0
            for idx, (name, symbol) in enumerate(SYMBOL_OPTIONS):
                if symbol == current_symbol:
                    current_symbol_index = idx
                    break

            selected_symbol_index = st.selectbox(
                f"Symbol {i+1}",
                range(len(SYMBOL_OPTIONS)),
                format_func=lambda x: SYMBOL_OPTIONS[x][0],
                index=current_symbol_index,
                help="Choose symbol for this comparison point",
                key=f"comparison_symbol_{i}",
            )
            st.session_state.comparison_files[i]["symbol"] = SYMBOL_OPTIONS[
                selected_symbol_index
            ][1]

        with col4:
            custom_name = st.text_input(
                f"Display name {i+1}",
                value=st.session_state.comparison_files[i]["name"],
                placeholder="e.g., SD Default",
                help="Custom name for this comparison point",
                key=f"comparison_name_{i}",
            )
            st.session_state.comparison_files[i]["name"] = custom_name

    temp_folder = None

    if uploaded_json:
        try:
            json_content = uploaded_json.read().decode("utf-8")
            results_data = json.loads(json_content)

            def validate_results_json(data):
                """Validate if uploaded JSON has the expected structure"""
                required_keys = ["topic", "configurations", "summary"]

                if not all(key in data for key in required_keys):
                    return False, f"Missing required keys. Expected: {required_keys}"

                if not isinstance(data["configurations"], dict):
                    return False, "Configurations must be a dictionary"

                for config_key, config_data in data["configurations"].items():
                    if not isinstance(config_data, dict):
                        return (
                            False,
                            f"Invalid configuration structure for: {config_key}",
                        )

                    required_config_keys = ["images", "aggregates"]
                    if not all(key in config_data for key in required_config_keys):
                        return (
                            False,
                            f"Missing required keys in configuration {config_key}: {required_config_keys}",
                        )

                return True, "Valid"

            def detect_metrics_from_json(results):
                """Detect which metrics are present in the uploaded results"""
                metrics = []

                if results["configurations"]:
                    first_config = list(results["configurations"].values())[0]
                    aggregates = first_config.get("aggregates", {})

                    if "avg_clip_score" in aggregates:
                        metrics.append("clip_score")
                    if "entropy_fairness" in aggregates:
                        metrics.append("entropy_fairness")
                    if "kl_fairness" in aggregates:
                        metrics.append("kl_fairness")

                return metrics

            is_valid, validation_message = validate_results_json(results_data)

            if is_valid:
                comparison_data = []
                for i, comp_file_info in enumerate(st.session_state.comparison_files):
                    if comp_file_info["file"] is not None:
                        try:
                            comp_json_content = (
                                comp_file_info["file"].read().decode("utf-8")
                            )
                            comp_data = json.loads(comp_json_content)

                            comp_is_valid, comp_validation_message = (
                                validate_results_json(comp_data)
                            )
                            if not comp_is_valid:
                                st.warning(
                                    f"⚠️ Comparison file {i+1} has invalid format: {comp_validation_message}. Skipping."
                                )
                                continue

                            comp_file_info["data"] = comp_data
                            comparison_data.append(comp_file_info)

                        except json.JSONDecodeError as e:
                            st.error(
                                f"❌ Error loading comparison file {i+1}: Invalid JSON format {str(e)}"
                            )
                            continue
                        except Exception as e:
                            st.error(
                                f"❌ Error processing comparison file {i+1}: {str(e)}"
                            )
                            continue

                detected_metrics = detect_metrics_from_json(results_data)

                current_topic = results_data.get("topic", "")
                current_configs = len(results_data.get("configurations", {}))

                data_id = f"{current_topic}_{current_configs}_{len(detected_metrics)}"

                if (
                    not hasattr(st.session_state, "current_data_id")
                    or st.session_state.current_data_id != data_id
                ):
                    st.session_state.results = results_data
                    st.session_state.comparison_data = comparison_data
                    st.session_state.selected_metrics = detected_metrics
                    st.session_state.all_available_metrics = detected_metrics
                    st.session_state.current_data_id = data_id

                    st.success("✅ JSON file loaded successfully!")
                    st.rerun()
                else:
                    st.success("✅ JSON file loaded successfully!")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Topic", results_data["topic"])
                with col2:
                    st.metric(
                        "Configurations", results_data["summary"]["total_configs"]
                    )
                with col3:
                    st.metric("Total Images", results_data["summary"]["total_images"])
                with col4:
                    st.metric("Available Metrics", len(detected_metrics))

                if comparison_data:
                    st.success(
                        f"✅ Successfully loaded {len(comparison_data)} comparison file(s)"
                    )
                    for comp in comparison_data:
                        name = (
                            comp["name"]
                            if comp["name"]
                            else f"Comparison {comparison_data.index(comp)+1}"
                        )
                        st.info(
                            f"🔶 {name}: {len(comp['data']['configurations'])} configurations"
                        )

                if detected_metrics:
                    metric_names = []
                    if "clip_score" in detected_metrics:
                        metric_names.append("🎯 CLIP Score")
                    if "entropy_fairness" in detected_metrics:
                        metric_names.append("⚖️ Normalized Entropy")
                    if "kl_fairness" in detected_metrics:
                        metric_names.append("📊 KL Divergence")

                else:
                    st.warning("⚠️ No recognized metrics found in the JSON file")

                temp_folder = "json_uploaded"

            else:
                st.error(f"❌ Invalid JSON file: {validation_message}")
                st.info(
                    "Please upload a valid results JSON file generated by this tool."
                )

        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {e}")
        except Exception as e:
            st.error(f"❌ Error reading JSON file: {e}")

if temp_folder and temp_folder != "json_uploaded" and selected_metrics:
    st.header("🚀 Processing")

    fairness_selected = (
        "entropy_fairness" in selected_metrics or "kl_fairness" in selected_metrics
    )
    if fairness_selected and not api_key:
        st.warning(
            "⚠️ **API Key Required**: You've selected Fairness Metrics which require an API key for demographic analysis. Please provide an API key in the sidebar."
        )
        st.info(
            "💡 **Tip**: You can still run CLIP Score evaluation without an API key if you deselect Fairness Metrics."
        )

    metrics_info = []
    if "clip_score" in selected_metrics:
        metrics_info.append("✅ **CLIP Score** - Alingment evaluation")
    if "entropy_fairness" in selected_metrics:
        if api_key:
            metrics_info.append(
                "✅ **Normalized Entropy (Fairness)** - Demographic diversity analysis"
            )
        else:
            metrics_info.append(
                "❌ **Normalized Entropy (Fairness)** - Requires API key"
            )
    if "kl_fairness" in selected_metrics:
        if api_key:
            metrics_info.append(
                "✅ **KL Divergence (Fairness)** - Demographic uniformity analysis"
            )
        else:
            metrics_info.append("❌ **KL Divergence (Fairness)** - Requires API key")

    if metrics_info:
        st.info("**Selected Metrics:**\n" + "\n".join(metrics_info))

    button_disabled = fairness_selected and not api_key
    button_text = "Start Evaluation" if not button_disabled else "API Key Required"

    if st.button(button_text, type="primary", disabled=button_disabled):
        try:
            provider_mapping = {
                "OpenAI": "openai",
                "OpenRouter": "openai",
                "Google Gemini": "gemini",
            }

            evaluator = ImageEvaluator(
                api_key=api_key,
                model_provider=provider_mapping.get(api_provider, "openai"),
                model_name=model_name,
                gpu_index=selected_gpu_index,
            )

            progress_bar = st.progress(0)
            status_text = st.empty()

            gpu_monitor_placeholder = None
            if selected_gpu_index is not None:
                gpu_monitor_placeholder = st.empty()

            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Processing... {progress:.1%} complete")

                if gpu_monitor_placeholder and selected_gpu_index is not None:
                    current_gpu_info = get_gpu_info()
                    if current_gpu_info and len(current_gpu_info) > selected_gpu_index:
                        gpu = current_gpu_info[selected_gpu_index]
                        gpu_monitor_placeholder.info(
                            f"""
                        **GPU {selected_gpu_index} Status:**
                        🔥 Utilization: {gpu['utilization']}% | 
                        💾 Memory: {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB | 
                        🆓 Free: {gpu['memory_free']:.0f} MB
                        """
                        )

            with st.spinner("Processing images..."):
                results = evaluator.process_images(
                    temp_folder, selected_metrics, progress_callback=update_progress
                )

            if gpu_monitor_placeholder:
                gpu_monitor_placeholder.empty()

            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")

            if "error" in results:
                st.error(results["error"])
            else:
                st.session_state.results = results
                st.session_state.selected_metrics = selected_metrics

                st.success("🎉 Evaluation completed successfully!")

        except Exception as e:
            st.error(f"Error during processing: {e}")

if hasattr(st.session_state, "results") and st.session_state.results:

    results = st.session_state.results
    selected_metrics = st.session_state.selected_metrics

    display_metrics = getattr(st.session_state, "display_metrics", selected_metrics)

    if hasattr(st.session_state, "all_available_metrics"):
        metric_display_names = {
            "clip_score": "🎯 CLIP Score",
            "entropy_fairness": "⚖️ Normalized Entropy",
            "kl_fairness": "📊 KL Divergence",
        }
        displayed_names = [metric_display_names.get(m, m) for m in display_metrics]
        st.info(f"**Displaying metrics:** {', '.join(displayed_names)}")

    st.subheader("Configuration Results")

    col1, col2 = st.columns([3, 1])
    with col1:
        custom_main_label = st.text_input(
            "Custom label for your configurations in chart:",
            value="Evaluated Configurations",
            placeholder="e.g., Fair Diffusion Configurations, Decodi Configurations, etc.",
            help="This text will appear in the chart legend instead of 'Evaluated Configurations'",
        )

        custom_pareto_label = st.text_input(
            "Custom label for Pareto-optimal configurations:",
            value="Pareto-Optimal Configurations",
            placeholder="e.g., Fair Diffusion Pareto-Optimal, Best Trade-offs, etc.",
            help="This text will appear in the chart legend for Pareto-optimal points",
        )
    with col2:
        st.write("")

    st.subheader("🎨 Chart Appearance")
    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        st.write("**Main Configurations:**")
        main_color_index = st.selectbox(
            "Color for your configurations:",
            range(len(COLOR_OPTIONS)),
            format_func=lambda x: COLOR_OPTIONS[x][0],
            index=5,
            help="Choose the color for your main configuration points",
            key="main_color_selector",
        )
        main_config_color = COLOR_OPTIONS[main_color_index][1]

        main_symbol_index = st.selectbox(
            "Symbol for your configurations:",
            range(len(SYMBOL_OPTIONS)),
            format_func=lambda x: SYMBOL_OPTIONS[x][0],
            index=0,
            help="Choose the symbol for your main configuration points",
            key="main_symbol_selector",
        )
        main_config_symbol = SYMBOL_OPTIONS[main_symbol_index][1]

    with col2:
        st.write("**Pareto-Optimal Points:**")
        pareto_color_index = st.selectbox(
            "Color for Pareto-optimal points:",
            range(len(COLOR_OPTIONS)),
            format_func=lambda x: COLOR_OPTIONS[x][0],
            index=1,
            help="Choose the color for Pareto-optimal points",
            key="pareto_color_selector",
        )
        pareto_config_color = COLOR_OPTIONS[pareto_color_index][1]

        pareto_symbol_index = st.selectbox(
            "Symbol for Pareto-optimal points:",
            range(len(SYMBOL_OPTIONS)),
            format_func=lambda x: SYMBOL_OPTIONS[x][0],
            index=7,
            help="Choose the symbol for Pareto-optimal points",
            key="pareto_symbol_selector",
        )
        pareto_config_symbol = SYMBOL_OPTIONS[pareto_symbol_index][1]

    with col3:
        st.write("**Axis Range Controls:**")
        min_y = st.number_input(
            "Min Y-axis value:",
            value=-0.1,
            step=0.1,
            format="%.2f",
            help="Set the minimum value for the Y-axis",
            key="min_y_axis",
        )

        max_y = st.number_input(
            "Max Y-axis value:",
            value=1.1,
            step=0.1,
            format="%.2f",
            help="Set the maximum value for the Y-axis",
            key="max_y_axis",
        )

        st.markdown("**Preview:**")
        st.markdown(
            f"Main: {COLOR_OPTIONS[main_color_index][0]} {SYMBOL_OPTIONS[main_symbol_index][0]}"
        )
        st.markdown(
            f"Pareto: {COLOR_OPTIONS[pareto_color_index][0]} {SYMBOL_OPTIONS[pareto_symbol_index][0]}"
        )
        st.markdown(f"Y-range: [{min_y:.2f}, {max_y:.2f}]")

    st.markdown("---")

    show_details = st.checkbox(
        "Show detailed demographic breakdowns",
        value=False,
        help="Show individual gender, ethnicity, and age metrics",
    )

    fairness_metric_for_chart = None
    if len(display_metrics) == 2:
        has_entropy = "entropy_fairness" in display_metrics
        has_kl = "kl_fairness" in display_metrics

        if has_entropy or has_kl:
            st.subheader("🎯 Chart Metric Selection")

            fairness_options = {}

            if has_entropy:
                fairness_options.update(
                    {
                        "entropy_fairness": "Overall Normalized Entropy",
                        "gender_entropy": "Gender Entropy",
                        "ethnicity_entropy": "Ethnicity Entropy",
                        "age_entropy": "Age Entropy",
                    }
                )

            if has_kl:
                fairness_options.update(
                    {
                        "kl_fairness": "Overall KL Divergence",
                        "gender_kl": "Gender KL Divergence",
                        "ethnicity_kl": "Ethnicity KL Divergence",
                        "age_kl": "Age KL Divergence",
                    }
                )

            default_metric = "entropy_fairness" if has_entropy else "kl_fairness"

            fairness_metric_for_chart = st.selectbox(
                "Select specific fairness metric for chart visualization:",
                options=list(fairness_options.keys()),
                format_func=lambda x: fairness_options[x],
                index=list(fairness_options.keys()).index(default_metric),
                help="Choose which specific fairness metric to display in the Pareto chart. This will replace the overall fairness metric.",
            )

            st.session_state.chart_fairness_metric = fairness_metric_for_chart

    summary_data = []
    for config_key, config_data in results["configurations"].items():
        row = {"Configuration": config_key, "Images": config_data["images"]}

        aggregates = config_data["aggregates"]

        if "clip_score" in display_metrics:
            row["CLIP Score"] = f"{aggregates.get('avg_clip_score', 0):.3f}"

        if "entropy_fairness" in display_metrics:
            row["Normalized Entropy"] = f"{aggregates.get('entropy_fairness', 0):.3f}"
            if show_details:
                row["Gender Entropy"] = f"{aggregates.get('gender_entropy', 0):.3f}"
                row["Ethnicity Entropy"] = (
                    f"{aggregates.get('ethnicity_entropy', 0):.3f}"
                )
                row["Age Entropy"] = f"{aggregates.get('age_entropy', 0):.3f}"

        if "kl_fairness" in display_metrics:
            kl_value = aggregates.get("kl_fairness", float("inf"))
            row["KL Divergence"] = (
                f"{kl_value:.3f}" if kl_value != float("inf") else "∞"
            )
            if show_details:
                gender_kl = aggregates.get("gender_kl", float("inf"))
                ethnicity_kl = aggregates.get("ethnicity_kl", float("inf"))
                age_kl = aggregates.get("age_kl", float("inf"))
                row["Gender KL"] = (
                    f"{gender_kl:.3f}" if gender_kl != float("inf") else "∞"
                )
                row["Ethnicity KL"] = (
                    f"{ethnicity_kl:.3f}" if ethnicity_kl != float("inf") else "∞"
                )
                row["Age KL"] = f"{age_kl:.3f}" if age_kl != float("inf") else "∞"

        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True)

    if len(display_metrics) == 2:
        st.subheader("📈 Fairness Trade-offs via Pareto Frontiers")

        chart_fairness_metric = None
        if "entropy_fairness" in display_metrics or "kl_fairness" in display_metrics:
            chart_fairness_metric = getattr(
                st.session_state, "chart_fairness_metric", None
            )

            if chart_fairness_metric:
                metric_names = {
                    "entropy_fairness": "⚖️ Overall Normalized Entropy",
                    "gender_entropy": "👫 Gender Entropy",
                    "ethnicity_entropy": "🌍 Ethnicity Entropy",
                    "age_entropy": "👶👨👴 Age Entropy",
                    "kl_fairness": "📊 Overall KL Divergence",
                    "gender_kl": "👫 Gender KL Divergence",
                    "ethnicity_kl": "🌍 Ethnicity KL Divergence",
                    "age_kl": "👶👨👴 Age KL Divergence",
                }
                metric_name = metric_names.get(
                    chart_fairness_metric, chart_fairness_metric
                )

        chart_data = create_pareto_chart_data(
            results,
            display_metrics,
            fairness_metric_override=chart_fairness_metric,
            comparison_data=getattr(st.session_state, "comparison_data", None),
        )

        if chart_data:
            plot_df = pd.DataFrame(chart_data)

            main_points = plot_df[~plot_df["is_comparison"]]
            comparison_points = plot_df[plot_df["is_comparison"]]

            fig = px.scatter(
                main_points,
                x="metric_0",
                y="metric_1",
                hover_data=["config"],
                labels={
                    "metric_0": (
                        plot_df.iloc[0]["metric_0_name"]
                        if len(plot_df) > 0
                        else "Metric 1"
                    ),
                    "metric_1": (
                        plot_df.iloc[0]["metric_1_name"]
                        if len(plot_df) > 0
                        else "Metric 2"
                    ),
                },
                color_discrete_sequence=[main_config_color],
            )

            fig.update_traces(
                marker=dict(
                    size=30,
                    symbol=main_config_symbol,
                    color=main_config_color,
                    line=dict(width=1, color="white"),
                ),
                name=custom_main_label,
                showlegend=True,
            )

            if len(comparison_points) > 0:
                comparison_data_with_colors = getattr(
                    st.session_state, "comparison_data", []
                )

                if comparison_data_with_colors:
                    points_processed = 0

                    for comp_idx, comp_info in enumerate(comparison_data_with_colors):
                        if comp_info.get("data") is not None:
                            num_configs = len(
                                comp_info["data"].get("configurations", {})
                            )

                            start_idx = points_processed
                            end_idx = start_idx + num_configs
                            comp_points = comparison_points.iloc[start_idx:end_idx]

                            if len(comp_points) > 0:
                                comp_color = comp_info.get("color", "#FF8C00")
                                comp_symbol = comp_info.get("symbol", "square")
                                comp_name = comp_info.get(
                                    "name", f"Comparison {comp_idx+1}"
                                )

                                fig.add_scatter(
                                    x=comp_points["metric_0"],
                                    y=comp_points["metric_1"],
                                    mode="markers",
                                    marker=dict(
                                        size=30,
                                        color=comp_color,
                                        symbol=comp_symbol,
                                        line=dict(width=2, color="white"),
                                    ),
                                    name=(
                                        comp_name
                                        if comp_name
                                        else f"Reference {comp_idx+1}"
                                    ),
                                    text=comp_points["config"],
                                    hovertemplate="<b>%{text}</b><br>"
                                    + f'{plot_df.iloc[0]["metric_0_name"] if len(plot_df) > 0 else "Metric 1"}: %{{x}}<br>'
                                    + f'{plot_df.iloc[0]["metric_1_name"] if len(plot_df) > 0 else "Metric 2"}: %{{y}}<br>'
                                    + "<extra></extra>",
                                    showlegend=True,
                                )

                            points_processed = end_idx
                else:
                    fig.add_scatter(
                        x=comparison_points["metric_0"],
                        y=comparison_points["metric_1"],
                        mode="markers",
                        marker=dict(
                            size=30, color="orange", line=dict(width=2, color="white")
                        ),
                        name="Reference Baseline",
                        text=comparison_points["config"],
                        hovertemplate="<b>%{text}</b><br>"
                        + f'{plot_df.iloc[0]["metric_0_name"] if len(plot_df) > 0 else "Metric 1"}: %{{x}}<br>'
                        + f'{plot_df.iloc[0]["metric_1_name"] if len(plot_df) > 0 else "Metric 2"}: %{{y}}<br>'
                        + "<extra></extra>",
                        showlegend=True,
                    )

            main_points_list = [
                (row["metric_0"], row["metric_1"]) for _, row in main_points.iterrows()
            ]

            pareto_indices = []
            for i, (x1, y1) in enumerate(main_points_list):
                is_pareto = True
                for j, (x2, y2) in enumerate(main_points_list):
                    if i != j and x2 >= x1 and y2 >= y1 and (x2 > x1 or y2 > y1):
                        is_pareto = False
                        break
                if is_pareto:
                    pareto_indices.append(i)

            if len(pareto_indices) > 1:
                pareto_points = [
                    (main_points.iloc[i]["metric_0"], main_points.iloc[i]["metric_1"])
                    for i in pareto_indices
                ]
                pareto_points.sort(key=lambda point: point[0])

                # Draw Pareto frontier line
                fig.add_scatter(
                    x=[point[0] for point in pareto_points],
                    y=[point[1] for point in pareto_points],
                    mode="lines+markers",
                    line=dict(color=pareto_config_color, width=3, dash="dash"),
                    marker=dict(
                        size=36,
                        color=pareto_config_color,
                        symbol=pareto_config_symbol,
                        line=dict(width=2, color="white"),
                    ),
                    name=custom_pareto_label,
                    showlegend=True,
                )

            elif len(pareto_indices) == 1:
                fig.add_scatter(
                    x=[main_points.iloc[pareto_indices[0]]["metric_0"]],
                    y=[main_points.iloc[pareto_indices[0]]["metric_1"]],
                    mode="markers",
                    marker=dict(
                        size=36,
                        color=pareto_config_color,
                        symbol=pareto_config_symbol,
                        line=dict(width=2, color="white"),
                    ),
                    name=custom_pareto_label,
                    showlegend=True,
                )

            all_x_values = list(plot_df["metric_0"])
            all_y_values = list(plot_df["metric_1"])
            data_max_x = max(all_x_values) if all_x_values else 1
            data_max_y = max(all_y_values) if all_y_values else 1

            fig.update_layout(
                height=800,
                width=1200,
                margin=dict(l=180, r=80, t=200, b=150),
                xaxis=dict(
                    range=[0.2, 0.242],
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor="lightgray",
                    title=dict(font=dict(size=52, color="black"), standoff=45),
                    tickfont=dict(size=44, color="black"),
                ),
                yaxis=dict(
                    range=[min_y, max_y],
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor="lightgray",
                    title=dict(font=dict(size=52, color="black"), standoff=50),
                    tickfont=dict(size=44, color="black"),
                ),
                legend=dict(
                    orientation="h",
                    x=0.5,
                    y=1.15,
                    xanchor="center",
                    yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    itemwidth=55,
                    itemsizing="constant",
                    font=dict(size=44, color="black"),
                    tracegroupgap=10,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("💾 Download Chart")
            col1, col2 = st.columns([3, 1])

            with col1:
                chart_filename = st.text_input(
                    "Chart filename (without extension):",
                    value="pareto_fairness_utility_tradeoff",
                    placeholder="Enter filename for the chart",
                    help="Enter the name for the downloaded SVG file (extension will be added automatically)",
                )

            with col2:
                st.write("")

                try:
                    svg_fig = go.Figure(fig)

                    svg_fig.update_layout(
                        margin=dict(l=500, r=100, t=360, b=180),
                        xaxis=dict(
                            range=[0.2, 0.242],
                            zeroline=True,
                            zerolinewidth=2,
                            zerolinecolor="lightgray",
                            title=dict(font=dict(size=65, color="black"), standoff=60),
                            tickfont=dict(size=55, color="black"),
                        ),
                        yaxis=dict(
                            range=[min_y, max_y],
                            zeroline=True,
                            zerolinewidth=2,
                            zerolinecolor="lightgray",
                            title=dict(
                                text=(
                                    plot_df.iloc[0]["metric_1_name"]
                                    if len(plot_df) > 0
                                    else "Metric 2"
                                ),
                                font=dict(size=65, color="black"),
                                standoff=140,
                            ),
                            tickfont=dict(size=55, color="black"),
                        ),
                        legend=dict(
                            orientation="h",
                            x=0.5,
                            y=1.24,
                            xanchor="center",
                            yanchor="bottom",
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="rgba(0,0,0,0.3)",
                            borderwidth=2,
                            itemwidth=55,
                            itemsizing="constant",
                            font=dict(size=55, color="black"),
                            tracegroupgap=10,
                        ),
                        width=2800,
                        height=1700,
                    )

                    svg_data = svg_fig.to_image(format="svg", width=2800, height=1700)

                    st.download_button(
                        label="📊 Download SVG",
                        data=svg_data,
                        file_name=(
                            f"{chart_filename}.svg"
                            if chart_filename
                            else "pareto_fairness_utility_tradeoff.svg"
                        ),
                        mime="image/svg+xml",
                        help="Download the chart as an SVG file",
                    )

                except Exception as e:
                    st.error(f"⚠️ SVG export failed: {str(e)}")

                    if "kaleido" in str(e).lower():
                        st.info("💡 Install with: `pip install kaleido`")
                    else:
                        st.info(
                            "💡 Try refreshing the page or check your chart configuration"
                        )

                    html_data = fig.to_html(include_plotlyjs=True)

                    st.download_button(
                        label="📄 Download HTML",
                        data=html_data,
                        file_name=(
                            f"{chart_filename}.html"
                            if chart_filename
                            else "pareto_fairness_utility_tradeoff.html"
                        ),
                        mime="text/html",
                        help="Download the chart as an interactive HTML file",
                    )

            st.markdown("---")

            st.subheader("🏆 Pareto Optimal Configurations")

            metric_0_name = (
                plot_df.iloc[0]["metric_0_name"] if len(plot_df) > 0 else "Metric 1"
            )
            metric_1_name = (
                plot_df.iloc[0]["metric_1_name"] if len(plot_df) > 0 else "Metric 2"
            )

            for i in pareto_indices:
                config = main_points.iloc[i]["config"]
                x_value = main_points.iloc[i]["metric_0"]
                y_value = main_points.iloc[i]["metric_1"]
                st.success(f"✅ **{config}** - ({x_value:.3f}, {y_value:.3f})")
                st.caption(
                    f"   📊 {metric_0_name}: {x_value:.3f} | {metric_1_name}: {y_value:.3f}"
                )

            if len(comparison_points) > 0:
                st.subheader("� Comparison Points")
                comparison_data_with_colors = getattr(
                    st.session_state, "comparison_data", []
                )

                if comparison_data_with_colors:
                    for comp_idx, comp_info in enumerate(comparison_data_with_colors):
                        comp_name = comp_info.get("name", f"Comparison {comp_idx+1}")
                        comp_color = comp_info.get("color", "#FF8C00")
                        comp_symbol = comp_info.get("symbol", "square")

                        color_emoji = "🔶"
                        for color_name, color_hex in COLOR_OPTIONS:
                            if color_hex == comp_color:
                                color_emoji = color_name.split()[0]
                                break

                        symbol_emoji = "●"
                        for symbol_name, symbol_value in SYMBOL_OPTIONS:
                            if symbol_value == comp_symbol:
                                symbol_emoji = symbol_name.split()[0]
                                break

                        comp_configs = list(
                            comp_info["data"].get("configurations", {}).keys()
                        )
                        if comp_configs:
                            st.info(
                                f"{color_emoji} {symbol_emoji} **{comp_name}**: {len(comp_configs)} configurations"
                            )
                else:
                    for _, comp_point in comparison_points.iterrows():
                        st.info(
                            f"🔶 **{comp_point['config']}** - "
                            f"Metric 1: {comp_point['metric_0']:.3f}, "
                            f"Metric 2: {comp_point['metric_1']:.3f}"
                        )

    elif len(display_metrics) < 2:
        st.info(
            "💡 **Pareto Chart:** Select exactly 2 metrics in the sidebar to see the Pareto frontier analysis."
        )
    elif len(display_metrics) > 2:
        st.info(
            "💡 **Pareto Chart:** Pareto analysis works with exactly 2 metrics. Currently showing all selected metrics in the table above."
        )

    st.subheader("💾 Download Results")

    results_json = json.dumps(results, indent=2)
    st.download_button(
        label="Download Results (JSON)",
        data=results_json,
        file_name=f"{results['topic']}_evaluation_results.json",
        mime="application/json",
    )

    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download Summary (CSV)",
        data=csv_data,
        file_name=f"{results['topic']}_summary.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    """
**Requirements for image filenames:**
- Format: `topic_parameters_seedX.extension`
- Example: `firefighter_cfg3.0_sld1000.0_seed1.png`
- All images should be of the same topic (e.g., all firefighters)

**Chart Customization Features:**
- 🎨 **Colors & Symbols**: Customize colors and symbols for main configurations, Pareto-optimal points, and comparison files
- 🏷️ **Custom Labels**: Set custom names for your data series in the chart legend
- 📊 **Flexible Comparison**: Add multiple comparison files, each with unique styling

"""
)

if temp_folder and temp_folder != "json_uploaded" and os.path.exists(temp_folder):
    try:
        shutil.rmtree(temp_folder)
    except Exception:
        pass
