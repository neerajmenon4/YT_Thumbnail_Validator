"use client";

import { useState } from "react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";

interface ThumbnailAnalysisResult {
  dimensions: {
    width: number;
    height: number;
    aspect_ratio: number;
    is_1280x720: boolean;
    is_16_9_ratio: boolean;
  };
  brightness: {
    brightness_value: number;
    brightness_percentage: number;
    is_bright: boolean;
    brightness_category: string;
  };
  text: {
    word_count: number;
    is_concise: boolean;
    readability_score: string;
    detected_text?: string;
  };
  color_palette: Array<{
    color: [number, number, number];
    percentage: number;
  }>;
  visualization: string;
}

interface LogoDetectionResult {
  is_present: boolean;
  accuracy: number;
  accuracy_percentage: number;
  template_matching?: {
    score: number;
    location: [number, number];
    dimensions: [number, number];
  };
  feature_matching?: {
    match_ratio: number;
    num_matches: number;
  };
  suggestions: string[];
  visualization: string;
  detection_method?: string;
  llm_response?: string;
}

export default function Home() {
  const [thumbnailFile, setThumbnailFile] = useState<File | null>(null);
  const [logoFile, setLogoFile] = useState<File | null>(null);
  const [thumbnailPreview, setThumbnailPreview] = useState<string | null>(null);
  const [logoPreview, setLogoPreview] = useState<string | null>(null);
  const [thumbnailAnalysis, setThumbnailAnalysis] = useState<ThumbnailAnalysisResult | null>(null);
  const [logoDetection, setLogoDetection] = useState<LogoDetectionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [apiKey, setApiKey] = useState<string>("");
  const [useGemini, setUseGemini] = useState<boolean>(false);

  const handleThumbnailChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setThumbnailFile(file);
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setThumbnailPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleLogoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setLogoFile(file);
      
      const reader = new FileReader();
      reader.onload = (e) => {
        setLogoPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeThumbnail = async () => {
    if (!thumbnailFile) {
      toast.error("Please select a thumbnail image");
      return;
    }

    setIsAnalyzing(true);
    try {
      const formData = new FormData();
      formData.append("thumbnail", thumbnailFile);

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/analyze-thumbnail/`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      setThumbnailAnalysis(data);
      toast.success("Thumbnail analysis completed");
    } catch (error) {
      console.error("Error analyzing thumbnail:", error);
      toast.error("Failed to analyze thumbnail");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const detectLogo = async () => {
    if (!thumbnailFile || !logoFile) {
      toast.error("Please select both thumbnail and logo images");
      return;
    }

    if (useGemini && !apiKey) {
      toast.error("Please provide a Gemini API key to use Gemini detection");
      return;
    }

    setIsDetecting(true);
    try {
      const formData = new FormData();
      formData.append("thumbnail", thumbnailFile);
      formData.append("logo", logoFile);
      formData.append("use_gemini", useGemini.toString());
      
      // Only append API key if using Gemini
      if (useGemini && apiKey) {
        formData.append("api_key", apiKey);
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/detect-logo/`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      setLogoDetection(data);
      toast.success(`Logo detection completed using ${useGemini ? "Gemini AI" : "traditional CV"} method`);
    } catch (error) {
      console.error("Error detecting logo:", error);
      toast.error("Failed to detect logo");
    } finally {
      setIsDetecting(false);
    }
  };

  const rgbToHex = (rgb: [number, number, number]) => {
    return "#" + rgb.map(x => {
      const hex = x.toString(16);
      return hex.length === 1 ? "0" + hex : hex;
    }).join("");
  };

  return (
    <main className="container mx-auto py-8 px-4">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-2">ACTIVI Video Thumbnail Validator</h1>
        <p className="text-xl text-muted-foreground">
          Analyze YouTube thumbnails and detect logos for optimal performance
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        <Card>
          <CardHeader>
            <CardTitle>Thumbnail Analysis</CardTitle>
            <CardDescription>
              Upload a YouTube thumbnail to analyze its properties
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex flex-col space-y-2">
                <label htmlFor="thumbnail" className="text-sm font-medium">
                  Upload Thumbnail (1280x720 recommended)
                </label>
                <Input 
                  id="thumbnail" 
                  type="file" 
                  accept="image/*" 
                  onChange={handleThumbnailChange} 
                />
              </div>
              
              {thumbnailPreview && (
                <div className="mt-4">
                  <p className="text-sm font-medium mb-2">Preview:</p>
                  <div className="relative w-full h-48 border rounded overflow-hidden">
                    <Image 
                      src={thumbnailPreview} 
                      alt="Thumbnail preview" 
                      fill 
                      style={{ objectFit: "contain" }} 
                    />
                  </div>
                </div>
              )}
            </div>
          </CardContent>
          <CardFooter>
            <Button 
              onClick={analyzeThumbnail} 
              disabled={!thumbnailFile || isAnalyzing}
              className="w-full"
            >
              {isAnalyzing ? "Analyzing..." : "Analyze Thumbnail"}
            </Button>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Logo Detection</CardTitle>
            <CardDescription>
              Check if your logo is present in the thumbnail
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex flex-col space-y-2">
                <label htmlFor="logo" className="text-sm font-medium">
                  Upload Logo
                </label>
                <Input 
                  id="logo" 
                  type="file" 
                  accept="image/*" 
                  onChange={handleLogoChange} 
                />
              </div>
              
              {logoPreview && (
                <div className="mt-4">
                  <p className="text-sm font-medium mb-2">Preview:</p>
                  <div className="relative w-full h-32 border rounded overflow-hidden">
                    <Image 
                      src={logoPreview} 
                      alt="Logo preview" 
                      fill 
                      style={{ objectFit: "contain" }} 
                    />
                  </div>
                </div>
              )}
              
              <div className="pt-4 border-t">
                <h4 className="text-sm font-medium mb-3">Detection Options</h4>
                
                <div className="flex items-center space-x-2 mb-4">
                  <input
                    type="checkbox"
                    id="useGemini"
                    checked={useGemini}
                    onChange={(e) => setUseGemini(e.target.checked)}
                    className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
                  />
                  <label htmlFor="useGemini" className="text-sm">
                    Use Gemini AI for detection
                  </label>
                </div>
                
                {useGemini && (
                  <div className="flex flex-col space-y-2">
                    <label htmlFor="apiKey" className="text-sm font-medium">
                      Gemini API Key
                    </label>
                    <Input
                      id="apiKey"
                      type="password"
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      placeholder="Enter your Gemini API key"
                      className="font-mono"
                    />
                    <p className="text-xs text-muted-foreground">
                      Your API key is not stored and will be cleared on page refresh
                    </p>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
          <CardFooter>
            <Button 
              onClick={detectLogo} 
              disabled={!thumbnailFile || !logoFile || isDetecting}
              className="w-full"
            >
              {isDetecting ? "Detecting..." : "Detect Logo"}
            </Button>
          </CardFooter>
        </Card>
      </div>

      {thumbnailAnalysis && (
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Thumbnail Analysis Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-lg font-semibold mb-4">Analysis Details</h3>
                
                <div className="space-y-6">
                  <div>
                    <h4 className="text-md font-medium mb-2">Dimensions</h4>
                    <div className="space-y-1">
                      <p>Size: {thumbnailAnalysis.dimensions.width}x{thumbnailAnalysis.dimensions.height}</p>
                      <p>Aspect Ratio: {thumbnailAnalysis.dimensions.aspect_ratio.toFixed(2)}</p>
                      <p>Correct Size (1280x720): 
                        <span className={thumbnailAnalysis.dimensions.is_1280x720 ? "text-green-500" : "text-red-500"}>
                          {thumbnailAnalysis.dimensions.is_1280x720 ? "✓" : "✗"}
                        </span>
                      </p>
                      <p>Correct Ratio (16:9): 
                        <span className={thumbnailAnalysis.dimensions.is_16_9_ratio ? "text-green-500" : "text-red-500"}>
                          {thumbnailAnalysis.dimensions.is_16_9_ratio ? "✓" : "✗"}
                        </span>
                      </p>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-md font-medium mb-2">Brightness</h4>
                    <div className="space-y-1">
                      <p>Brightness: {thumbnailAnalysis.brightness.brightness_percentage.toFixed(1)}%</p>
                      <p>Category: {thumbnailAnalysis.brightness.brightness_category}</p>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-md font-medium mb-2">Text Readability</h4>
                    <div className="space-y-1">
                      <p>Word Count: {thumbnailAnalysis.text.word_count}</p>
                      <p>Is Concise: 
                        <span className={thumbnailAnalysis.text.is_concise ? "text-green-500" : "text-red-500"}>
                          {thumbnailAnalysis.text.is_concise ? "✓" : "✗"}
                        </span>
                      </p>
                      <p>Readability Score: {thumbnailAnalysis.text.readability_score}</p>
                      {thumbnailAnalysis.text.detected_text && (
                        <div>
                          <p className="font-medium mt-2">Detected Text:</p>
                          <p className="text-sm bg-muted p-2 rounded">{thumbnailAnalysis.text.detected_text}</p>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-md font-medium mb-2">Color Palette</h4>
                    <div className="flex flex-wrap gap-2">
                      {thumbnailAnalysis.color_palette.map((item, index) => (
                        <div key={index} className="flex flex-col items-center">
                          <div 
                            className="w-12 h-12 rounded-md border" 
                            style={{ backgroundColor: rgbToHex(item.color) }}
                          />
                          <span className="text-xs mt-1">{item.percentage.toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-4">Visualization</h3>
                <div className="border rounded overflow-hidden">
                  {thumbnailAnalysis.visualization ? (
                    <img 
                      src={`data:image/png;base64,${thumbnailAnalysis.visualization}`} 
                      alt="Analysis visualization" 
                      className="w-full h-auto" 
                    />
                  ) : (
                    <div className="p-4 text-center text-gray-500">
                      Visualization not available
                    </div>
                  )}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {logoDetection && (
        <Card>
          <CardHeader>
            <CardTitle>Logo Detection Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-lg font-semibold mb-4">Detection Details</h3>
                
                <div className="space-y-6">
                  <div>
                    <h4 className="text-md font-medium mb-2">Logo Presence</h4>
                    <div className="space-y-1">
                      <p>Status: 
                        <span className={logoDetection.is_present ? "text-green-500 font-medium" : "text-red-500 font-medium"}>
                          {logoDetection.is_present ? "Detected" : "Not Detected"}
                        </span>
                      </p>
                      <p>Accuracy: {logoDetection.accuracy_percentage.toFixed(1)}%</p>
                      <p>Detection Method: {logoDetection.detection_method === 'gemini' ? 'Gemini AI' : 'Traditional CV'}</p>
                    </div>
                  </div>
                  
                  {logoDetection.detection_method === 'traditional' && logoDetection.template_matching && (
                    <div className="mt-4">
                      <h4 className="text-md font-medium mb-2">Template Matching</h4>
                      <div className="space-y-1">
                        <p>Score: {logoDetection.template_matching.score.toFixed(3)}</p>
                        <p>Location: [{logoDetection.template_matching.location[0]}, {logoDetection.template_matching.location[1]}]</p>
                        <p>Dimensions: {logoDetection.template_matching.dimensions[0]}x{logoDetection.template_matching.dimensions[1]}</p>
                      </div>
                    </div>
                  )}
                
                {logoDetection.detection_method === 'traditional' && logoDetection.feature_matching && (
                    <div className="mt-4">
                      <h4 className="text-md font-medium mb-2">Feature Matching</h4>
                      <div className="space-y-1">
                        <p>Match Ratio: {logoDetection.feature_matching.match_ratio.toFixed(3)}</p>
                        <p>Number of Matches: {logoDetection.feature_matching.num_matches}</p>
                      </div>
                    </div>
                  )}
                
                  {logoDetection.detection_method === 'gemini' && (
                    <div className="mt-4">
                      <h4 className="text-md font-medium mb-2">Gemini AI Detection</h4>
                      <div className="space-y-1">
                        <p>Method: Gemini AI Vision</p>
                        <p>Confidence: {logoDetection.accuracy_percentage.toFixed(1)}%</p>
                      </div>
                    </div>
                  )}
                  
                  {logoDetection.suggestions && logoDetection.suggestions.length > 0 && (
                    <div className="mt-4">
                      <h4 className="text-md font-medium mb-2">Suggestions</h4>
                      <div className="bg-amber-50 border border-amber-200 rounded-md p-3">
                        <ul className="list-disc list-inside space-y-1 text-amber-800">
                          {logoDetection.suggestions.map((suggestion, index) => (
                            <li key={index}>{suggestion}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-4">Visualization</h3>
                <div className="border rounded overflow-hidden">
                  <img 
                    src={`data:image/png;base64,${logoDetection.visualization}`} 
                    alt="Logo detection visualization" 
                    className="w-full h-auto" 
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </main>
  );
}
