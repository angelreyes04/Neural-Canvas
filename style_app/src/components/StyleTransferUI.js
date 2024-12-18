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
  const colabUrl='https://5bcd-34-169-34-1.ngrok-free.app';

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
        setArtist(data.artist)
      }
      if (data.date) {
        setDate(data.date)
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

  useEffect(() => {
    console.log("Artist:", artist);
  }, [artist]);

  useEffect(() => {
    console.log("Date:", date);
  }, [date]);

  useEffect(() => {
    console.log("Style Prompt:", stylePrompt);
  }, [stylePrompt]);
  
  return (
    <div className="p-4 max-w-2x1 mx-auto">
      <h1 className="text-2xl font-bold mb-4 text-center bg-gradient-to-r text-transparent bg-clip-text from-red-500 via-yellow-500 to-blue-500">
        Neural Style Transfer
      </h1>

      <div className="mb-4">
        Input a style description and content image below!
      </div>

      <div className="mb-4">
        <input
            type="text"
            value={stylePrompt}
            onChange={(e) => setStylePrompt(e.target.value)}
            placeholder="Enter style description (Like 'happy, rainbow')"
            className="w-full p-2 border rounded"
          />
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="w-full p-2 border rounded mb-2"
          />
          {contentImage && (
            <img 
              src={URL.createObjectURL(contentImage)} 
              alt="Content" 
              className="w-full h-auto object-cover rounded"
            />
          )}
        </div>

        <div>
          {targetImage ? (
            <img 
              src={targetImage}
              alt="Target" 
              className="w-full h-auto object-cover rounded"
            />
          ): (
            <div className="w-full h-full border-2 border-dashed flex items-center justify-center text-gray-400 rounded">
              Target Image
            </div>
          )}  
        </div>
      </div>

      <div className="flex justify-center space-x-4 mb-4">
        <button
          onClick={handleSubmit}
          disabled={!contentImage || !stylePrompt || loading}
          className="bg-blue-500 text-white p-2 rounded disabled:opacity-50"
        >
          {loading ? 'Processing...' : 'Generate'}
        </button>
        <button
          onClick={handleRefresh}
          className="bg-gray-500 text-white p-2 rounded"
        >
          Refresh
        </button>
      </div>

      <div>
        <h2 className="font-bold mt-4">Style Images and Weights:</h2>
      </div>
      {styleImages.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          {styleImages.map((img, index) => (
            <div key={index} className ="border rounded p-2">
              <img
                src={img}
                alt={`Style '${stylePrompt[index]}' #${index+1}`}
                className="w-full h-80 object-cover rounded mb-2"
              />
              <div className="text-sm">
                <h3> Style {index+1} </h3>
                {(title?.[index] || artist?.[index] || date?.[index]) ? (
                  <p>
                    Art Title: {title?.[index] || "Unknown Title"}
                    {artist?.[index] ? ` by ${artist[index]}` : ""}
                    {date?.[index] ? `, ${date[index]}`: ""}
                  </p>
                ): null}
                <p>Medium: {medium?.[index] || "Unknown Medium"}</p>
                <p>Weight: {styleWeights[index]}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default StyleTransferUI;