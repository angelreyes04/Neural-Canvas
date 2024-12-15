import React, { useState } from 'react';

const StyleTransferUI = () => {
  const [contentImage, setContentImage] = useState(null);
  const [stylePrompt, setStylePrompt] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [colabUrl, setColabUrl] = useState('');

  useEffect(() => {
    const fetchColabUrl = async() => {
      try {
        const response = await fetch('/get-colab-url');
        const data = await response.json();
        setColabUrl(data.colab_url);
      }
      catch (error) {
        console.error("Error fetching colab public url", error);
      }
    };
    fetchColabUrl();
  }, []);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setContentImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const response = await fetch('${colabUrl}/style-transfer', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contentImage: contentImage,
          stylePrompt: stylePrompt
        })
      });
      
      const data = await response.json();
      setResult(data.result);

      const styleResponse = await fetch('${colabUrl}/get-similar-styles', {
        method: 'POST', 
        headers: {
          'Content-Type': 'application/json',  
        }, 
        body: JSON.stringify({
          stylePrompt:stylePrompt,
        }),
      });
      const similarImgs = await styleResponse.json();
      setSimilarImagePaths(similarImgs.imagePaths);
      setSimilarWeights(similarImgs.weights);
    } catch (error) {
      console.error('Error:', error);
    }
    setLoading(false);
  };

  return (
    <div className="p-4 max-w-md mx-auto">
      <h1 className="text-2xl font-bold mb-4 text-center">Style Transfer</h1>
      
      <div className="space-y-4">
        <div>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="w-full p-2 border rounded"
          />
          {contentImage && (
            <img 
              src={contentImage} 
              alt="Content" 
              className="mt-2 max-w-full h-auto"
            />
          )}
        </div>

        <div>
          <input
            type="text"
            value={stylePrompt}
            onChange={(e) => setStylePrompt(e.target.value)}
            placeholder="Enter style description"
            className="w-full p-2 border rounded"
          />
        </div>

        <button
          onClick={handleSubmit}
          disabled={!contentImage || !stylePrompt || loading}
          className="w-full bg-blue-500 text-white p-2 rounded disabled:opacity-50"
        >
          {loading ? 'Processing...' : 'Generate'}
        </button>

        {result && (
          <div>
            <h2 className="font-bold mt-4">Result:</h2>
            <img 
              src={result} 
              alt="Result" 
              className="mt-2 max-w-full h-auto"
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default StyleTransferUI;