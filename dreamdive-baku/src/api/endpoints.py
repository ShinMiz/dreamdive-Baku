from flask import Flask, request, jsonify
from models.dream_analyzer import DreamAnalyzer
from models.reframing_llm import ReframingLLM
from models.video_generator import VideoGenerator

app = Flask(__name__)

@app.route('/analyze_dream', methods=['POST'])
def analyze_dream():
    data = request.json
    dream_content = data.get('dream_content')
    
    analyzer = DreamAnalyzer()
    analysis_result = analyzer.analyze(dream_content)
    
    return jsonify(analysis_result)

@app.route('/reframe_dream', methods=['POST'])
def reframe_dream():
    data = request.json
    dream_content = data.get('dream_content')
    
    reframer = ReframingLLM()
    reframed_dream = reframer.reframe(dream_content)
    
    return jsonify({'reframed_dream': reframed_dream})

@app.route('/generate_video', methods=['POST'])
def generate_video():
    data = request.json
    dream_content = data.get('dream_content')
    
    video_gen = VideoGenerator()
    video_path = video_gen.generate(dream_content)
    
    return jsonify({'video_path': video_path})

if __name__ == '__main__':
    app.run(debug=True)