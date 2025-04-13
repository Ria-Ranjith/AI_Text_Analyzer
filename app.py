import gradio as gr
from transformers import pipeline

# Load summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def analyze_text(file, prompt):
    if not file and not prompt:
        return "Please upload a .txt file or enter a prompt.", "", None

    content = ""

    # If file is uploaded
    if file is not None:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

    # Combine both if available
    combined_input = (prompt + "\n\n" + content).strip()

    # Handle empty input
    if not combined_input:
        return "Input is empty. Please upload a file or enter text.", "", None

    # Limit input for model
    combined_input = combined_input[:1024]

    # Generate summary
    summary = summarizer(combined_input, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    word_count = len(summary.split())

    # Save result to file
    output_file = "summary_result.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Summary:\n{summary}\n\nWord Count: {word_count}")

    return summary, f"{word_count} words", output_file

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 AI-Powered Text Analyzer")
    gr.Markdown(
        "📝 *Enter a prompt or upload a .txt file* — or both — and click *Analyze* to generate an AI summary."
    )

    with gr.Row():
        with gr.Column():
            file_input = gr.File(file_types=[".txt"], label="📄 Upload .txt File")
            prompt_input = gr.Textbox(placeholder="e.g. Summarize this article", label="💬 Your Prompt")
            with gr.Row():
                analyze_btn = gr.Button("🚀 Analyze", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")

        with gr.Column():
            summary_output = gr.Textbox(label="📝 Summary", lines=10)
            word_count_output = gr.Textbox(label="🔢 Word Count")
            download_btn = gr.File(label="⬇️ Download Summary")

    analyze_btn.click(analyze_text, inputs=[file_input, prompt_input], outputs=[summary_output, word_count_output, download_btn])
    clear_btn.click(lambda: ("", "", None), outputs=[summary_output, word_count_output, download_btn])

demo.launch()
