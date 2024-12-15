from flask import Flask, request, jsonify

app = Flask(__name__)

COLAB_PUBLIC_URL = "https://ff09-128-148-128-30.ngrok-free.app"

@app.route('/style-transfer', methods=['POST'])
def style_transfer():
    #get image and style prompt from user input
    content_image = request.json['contentImage']
    style_descriptions = request.json['stylePrompt']

    # run style transfer function from colab notebook
    result = apply_style_transfer(
        content_image_path=content_image,
        style_descriptions=style_descriptions,
        output_path='src/images/content/output.jpg',
        num_styles=1,
        do_prune_model=False,
        do_quantize_model=True
    )

    # Return the result to the frontend
    return jsonify({'result': result})

@app.route('/get-similar-styles', methods=['POST'])
def get_similar_styles():
    style_prompt = request.json['stylePrompt']
    top_k_image_paths, top_k_weights = get_k_most_similar_image_paths_and_weights(style_prompt, k=3)
    return jsonify({'imagePaths': top_k_image_paths, 'weights': top_k_weights})

@app.route("/get-colab-url", methods=["GET"])
def get_colab_url():
    return jsonify({"colab_url": COLAB_PUBLIC_URL})

if __name__ == '__main__':
    app.run(debug=True)