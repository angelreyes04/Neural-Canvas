import React, { useState , useEffect } from 'react';

const StyleTransferUI = () => {
  const [contentImage, setContentImage] = useState(null);
  const [stylePrompt, setStylePrompt] = useState('');
  const [targetImage, setTargetImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const colabUrl='https://0bad-34-143-189-192.ngrok-free.app';

  const handleImageUpload = (event) => {
    //console.log("image upload event triggered")
    const file = event.target.files[0];
    if (file) {
      setContentImage(file)
    }
  };

  const handleSubmit = async () => {
    console.log('Generate button clicked');
    if(!contentImage) {
      console.log("Please upload an image");
      return;
    }
    if (!stylePrompt) {
      console.log("Please input a style prompt");
      return;
    }
    if (!colabUrl) {
      console.log("Colab URL not found");
      return;
    }
    
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('content_image', contentImage)
      formData.append('stylePrompt', stylePrompt)
      
      console.log('Sending request to:', `${colabUrl}/nst`);
      
      const response = await fetch(`${colabUrl}/nst`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok){
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Received data:", data);
      if (data.image_url) {
        setTargetImage(`${colabUrl}${data.image_url}`)
      }

      //const similarImgs = await styleResponse.json();
      //setSimilarImagePaths(similarImgs.imagePaths);
      //setSimilarWeights(similarImgs.weights);
    } catch (error) {
      console.error('Error:', error);
    }
    finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    console.log(`targetImage: ${targetImage}`);
  }, [targetImage]);

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
              src={URL.createObjectURL(contentImage)} 
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

        {targetImage && (
          <div>
            <h2 className="font-bold mt-4">Target Image:</h2>
            <img 
              src={targetImage}
              alt="Target" 
              className="mt-2 max-w-full h-auto"
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default StyleTransferUI;