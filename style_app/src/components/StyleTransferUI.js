import React, { useState , useEffect } from 'react';

const StyleTransferUI = () => {
  const [contentImage, setContentImage] = useState(null);
  const [stylePrompt, setStylePrompt] = useState('');
  const [targetImage, setTargetImage] = useState(null);
  const [styleImages, setStyleImages] = useState([])
  const [styleWeights, setStyleWeights] = useState([])
  const [loading, setLoading] = useState(false);
  const [title, setTitle] = useState([])
  const [artist, setArtist] = useState([])
  const [date, setDate] = useState([])
  const [medium, setMedium] = useState([])
  const [hasRefreshed, setHasRefreshed] = useState(false)
  const colabUrl='https://ed01-34-105-107-121.ngrok-free.app';

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
        body: formData,
        headers: {
          'ngrok-skip-browser-warning': 'true',
        }
      });

      if (!response.ok){
        throw new Error(`Error: ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Received data:", data);
      if (data.target_img) {
        setTargetImage(`data:image/jpeg;base64,${data.target_img}`)
      }
      if (data.style_imgs) {
        const styleImagesBase64 = data.style_imgs.map(img => `data:image/jpeg;base64,${img}`);
        setStyleImages(styleImagesBase64)
        setHasRefreshed(false)
      }
      if (data.style_weights) {
        setStyleWeights(data.style_weights)
      }
      if (data.art_title) {
        setTitle(data.art_title)
      }
      if (data.artist) {
        setArtist(data.setArtist)
      }
      if (data.date) {
        setDate(data.data)
      }
      if (data.medium) {
        setMedium(data.medium)
      }
    } catch (error) {
      console.error('Error:', error);
    }
    finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    setContentImage(null);
    setStylePrompt('');
    setTargetImage(null);
    setStyleImages([]);
    setStyleWeights([]);
    setTitle([]);
    setArtist([]);
    setDate([]);
    setMedium([]);
    setHasRefreshed(true); 
  }

  
  return (
    <div className="p-4 max-w-md mx-auto">
      <h1 className="text-2xl font-bold mb-4 text-center bg-gradient-to-r text-transparent bg-clip-text from-red-500 via-yellow-500 to-blue-500">
        Neural Style Transfer
      </h1>
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
            placeholder="Enter style description (Like 'happy, rainbow')"
            className="w-full p-2 border rounded"
          />
        </div>

        <button
          onClick={handleSubmit}
          disabled={(!contentImage || !stylePrompt || loading) && (targetImage && !hasRefreshed)}
          className="w-full bg-blue-500 text-white p-2 rounded disabled:opacity-50"
        >
          {loading ? 'Processing...' : 'Generate'}
        </button>
        <button
          onClick={handleRefresh}
          className="w-full bg-gray-500 text-white p-2 rounded mt-4"
        >
          Refresh
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

        {styleImages.length > 0 && (
          <div>
            <h2 className="font-bold mt-4">Style Images and Weights:</h2>
            {styleImages.map((img, index) => (
              <div key={index} className ="mb-4">
                <h3> Style {index+1} </h3>
                {(title?.[index] || artist?.[index] || date?.[index]) ? (
                  <p>
                    Art Title: {title?.[index] || "Unknown Title"}
                    {artist?.[index] ? `by ${artist[index]}` : ""}
                    {date?.[index] ? `, ${date[index]}`: ""}
                  </p>
                ): null}
                <p>Medium: {medium?.[index] || "Unknown Medium"}</p>
                <p>Weight: {styleWeights[index]}</p>

                <img
                  src={img}
                  alt={`Style ${index+1}`}
                  className="mt-2 max-w-full h-auto"
                  />
              </div>
          ))}
          </div>
        )}   
      </div>
    </div>
  );
};

export default StyleTransferUI;