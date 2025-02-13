from typing import Dict, Optional, List, Callable
from pathlib import Path
import logging
import yaml
import json
import pandas as pd
import logging
from openai import OpenAI


__all__ = ['OpenAIClient', 'MovieClassifier']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, config_path: str, agent_path: str):
        try:
            self.config_path = Path(config_path).resolve()
            self.agent_path = Path(agent_path).resolve()
            logger.info(f"Initializing OpenAI client with config: {self.config_path}")
            
            self.config = self._load_config()
            self.messages = self._load_messages()
            
            self.client = OpenAI(
                api_key=self.config["openai"]["api_key"],
                timeout=60.0
            )
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _load_config(self) -> Dict:
        try:
            logger.info(f"Loading configuration from {self.config_path}")
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
                
            with open(self.config_path, "r", encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
            if not config.get("openai", {}).get("api_key"):
                raise ValueError("OpenAI API key not found in config file")
                
            return config
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _load_messages(self) -> List[Dict]:
        try:
            logger.info(f"Loading messages from {self.agent_path}")
            if not self.agent_path.exists():
                raise FileNotFoundError(f"Agent file not found at {self.agent_path}")
                
            with open(self.agent_path, "r", encoding='utf-8') as file:
                data = yaml.safe_load(file)
                
            if not isinstance(data.get("messages"), list):
                raise ValueError("Messages not found or invalid format in agent file")
                
            return data["messages"]
            
        except Exception as e:
            logger.error(f"Error loading messages: {e}")
            raise
    
    def format_messages(self, variables: Dict[str, str]) -> List[Dict]:
        try:
            formatted_messages = []
            for message in self.messages:
                new_message = message.copy()
                if message["role"] == "user":
                    try:
                        new_message["content"] = message["content"].format(**variables)
                    except KeyError as e:
                        raise ValueError(f"Missing required variable in template: {e}")
                formatted_messages.append(new_message)
            return formatted_messages
            
        except Exception as e:
            logger.error(f"Error formatting messages: {e}")
            raise
    
    def get_completion(self, 
                      variables: Dict[str, str], 
                      model: str = "gpt-4o-mini",
                      temperature: float = 0,
                      max_tokens: Optional[int] = None) -> str:
        try:
            logger.info(f"Getting completion for model {model}")
            formatted_messages = self.format_messages(variables)
            
            params = {
                "model": model,
                "messages": formatted_messages,
                "temperature": temperature
            }
            if max_tokens:
                params["max_tokens"] = max_tokens
                
            logger.info("Sending request to OpenAI API")
            response = self.client.chat.completions.create(**params)
            
            if not response.choices:
                raise ValueError("No choices in OpenAI response")
                
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting completion: {e}")
            if "openai" in str(type(e)).lower():
                raise RuntimeError(f"OpenAI API error: {e}")
            raise

    def run(self, variables: Dict[str, str], model: str = "gpt-4o-mini",
            temperature: float = 0, max_tokens: Optional[int] = None) -> str:
        try:
            return self.get_completion(variables, model, temperature, max_tokens)
        except Exception as e:
            logger.error(f"Error running OpenAI client: {e}")
            raise

class MovieClassifier:
    def __init__(self, openai_client: 'OpenAIClient', classification_path: str, prompt_path: str):
        try:
            self.openai_client = openai_client
            self.classification_path = Path(classification_path).resolve()
            self.prompt_path = Path(prompt_path).resolve()
            logger.info(f"Initializing MovieClassifier with classification path: {self.classification_path}")
            
            self.categories = self._load_classification()
            self.prompt_template = self._load_prompt()
            
            logger.info("MovieClassifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MovieClassifier: {e}")
            raise
    
    def _load_classification(self) -> Dict:
        try:
            logger.info(f"Loading classification from {self.classification_path}")
            if not self.classification_path.exists():
                raise FileNotFoundError(f"Classification file not found at {self.classification_path}")
                
            with open(self.classification_path, "r", encoding='utf-8') as file:
                categories = yaml.safe_load(file)
                
            if not categories:
                raise ValueError("Classification dictionary is empty or invalid")
                
            return categories
            
        except Exception as e:
            logger.error(f"Error loading classification: {e}")
            raise
    
    def _load_prompt(self) -> Dict:
        try:
            logger.info(f"Loading prompt from {self.prompt_path}")
            if not self.prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found at {self.prompt_path}")
                
            with open(self.prompt_path, "r", encoding='utf-8') as file:
                prompt = yaml.safe_load(file)
                
            if not prompt:
                raise ValueError("Prompt template is empty or invalid")
                
            return prompt
            
        except Exception as e:
            logger.error(f"Error loading prompt: {e}")
            raise
    
    def format_variables(self, description: str) -> Dict[str, str]:
        try:
            logger.info("Formatting variables for OpenAI query")
            return {
                "description": description,
                "categorization_context": yaml.dump(self.categories, default_flow_style=False),
                "prompt_template": yaml.dump(self.prompt_template, default_flow_style=False)
            }
        except Exception as e:
            logger.error(f"Error formatting variables: {e}")
            raise
    
    def classify(self, 
                description: str,
                model: str = "gpt-4o-mini",
                temperature: float = 0,
                max_tokens: Optional[int] = 500) -> Dict:
        try:
            logger.info(f"Classifying movie with description: {description[:50]}...")
            variables = self.format_variables(description)
            
            logger.info("Sending classification request to OpenAI")
            result_json = self.openai_client.run(
                variables=variables,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            classification = json.loads(result_json)
            logger.info(f"Classification completed successfully")
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying movie: {e}")
            if "json" in str(type(e)).lower():
                raise ValueError(f"Invalid JSON response from OpenAI: {e}")
            raise

    def run(self, description: str, model: str = "gpt-4o-mini", 
            temperature: float = 0, max_tokens: int = 500) -> Dict:
        try:
            return self.classify(description, model, temperature, max_tokens)
        except Exception as e:
            logger.error(f"Error running MovieClassifier: {e}")
            raise

def extract_movie_data(response: Dict) -> Dict[str, any]:
    """
    Extracts movie classification data from the OpenAI response
    
    Args:
        response (Dict): Response dictionary from OpenAI
        
    Returns:
        Dict[str, any]: Dictionary with movie classification data
    """
    # Define required fields at the start
    required_fields = [
        "genero_principal",
        "subgeneros",
        "tono",
        "estilo_narrativo",
        "temas_clave",
        "influencias_cinematograficas",
        "audiencia_objetivo"
    ]
    
    try:
        # The response is already a dictionary, no need to parse JSON
        data = response
        
        # Convert lists to string representations for DataFrame compatibility
        for key in data:
            if isinstance(data[key], list):
                data[key] = ', '.join(data[key])
                
        # Check for missing fields and set them to "No especificado"
        for field in required_fields:
            if field not in data or not data[field]:
                data[field] = "No especificado"
                
        return data
        
    except Exception as e:
        logger.error(f"Error extracting movie data: {e}")
        return {field: "Error en procesamiento" for field in required_fields}

def process_movies_to_dataframe(
    df: pd.DataFrame,
    config_path: str,
    agent_path: str,
    movie_classification_path: str,
    prompt_classification_path: str,
    extract_fn: Callable = extract_movie_data,
    description_column: str = 'description'
) -> pd.DataFrame:
    """
    Process movie descriptions through the classifier and add results to DataFrame
    
    Args:
        df (pd.DataFrame): Input DataFrame with movie descriptions
        config_path (str): Path to OpenAI config file
        agent_path (str): Path to agent classification file
        movie_classification_path (str): Path to movie classification schema
        prompt_classification_path (str): Path to prompt classification file
        extract_fn (Callable): Function to extract data from response
        description_column (str): Name of the column containing movie descriptions
        
    Returns:
        pd.DataFrame: DataFrame with added classification columns
    """
    try:
        # Initialize clients
        openai_client = OpenAIClient(config_path, agent_path)
        movie_classifier = MovieClassifier(
            openai_client=openai_client,
            classification_path=movie_classification_path,
            prompt_path=prompt_classification_path
        )

        # Process each row
        for idx, row in df.iterrows():
            try:
                logger.info(f"Processing movie {idx + 1}/{len(df)}")
                
                # Get the movie description
                description = row[description_column]
                if pd.isna(description) or not description.strip():
                    logger.warning(f"Empty description found in row {idx}")
                    continue

                # Classify the movie
                result = movie_classifier.run(description)
                
                # Extract and add classification data to DataFrame
                classification_data = extract_fn(result)
                for key, value in classification_data.items():
                    df.at[idx, key] = value
                    
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                # Set error values for this row
                error_values = {
                    "genero_principal": "Error en procesamiento",
                    "subgeneros": "Error en procesamiento",
                    "tono": "Error en procesamiento",
                    "estilo_narrativo": "Error en procesamiento",
                    "temas_clave": "Error en procesamiento",
                    "influencias_cinematograficas": "Error en procesamiento",
                    "audiencia_objetivo": "Error en procesamiento"
                }
                for key, value in error_values.items():
                    df.at[idx, key] = value

        return df
        
    except Exception as e:
        logger.error(f"Error in process_movies_to_dataframe: {e}")
        raise