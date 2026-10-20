import axios from "axios";

const API = axios.create({
  baseURL: "http://localhost:8000",
});

export const predictPremium = async (data) => {
  const response = await API.post("/predict", data);
  return response.data;
};

export const askQuestion = async (question,history) => {
  const response = await API.post("/chat", {
    question: question,
    history: history,
  });

  return response.data;
};