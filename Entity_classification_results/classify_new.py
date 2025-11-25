import pandas as pd
import json
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import os
from os import path

class DeFiClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            lowercase=True
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
    def extract_features(self, entity_name):
        """Extract hand-crafted features from entity name"""
        features = {}
        
        # Rule-based features
        features['starts_with_at'] = 1 if entity_name.startswith('@') else 0
        features['has_swap'] = 1 if 'swap' in entity_name.lower() else 0
        features['has_dex'] = 1 if 'dex' in entity_name.lower() else 0
        features['has_exchange'] = 1 if 'exchange' in entity_name.lower() else 0
        features['has_protocol'] = 1 if 'protocol' in entity_name.lower() else 0
        features['has_finance'] = 1 if 'finance' in entity_name.lower() else 0
        features['has_aggregator_terms'] = 1 if any(term in entity_name.lower() 
                                                      for term in ['paraswap', '1inch', 'odos', 'aggregator']) else 0
        
        # CEX indicators
        cex_terms = ['binance', 'coinbase', 'kraken', 'bybit', 'okx', 'kucoin', 
                     'gate.io', 'bitfinex', 'bitstamp', 'crypto.com']
        features['is_known_cex'] = 1 if any(cex in entity_name.lower() for cex in cex_terms) else 0
        
        # DeFi protocol indicators
        defi_terms = ['uniswap', 'curve', 'aave', 'compound', 'sushi', 'pancake',
                      'balancer', 'velodrome', 'aerodrome', 'raydium', 'orca']
        features['is_known_defi'] = 1 if any(defi in entity_name.lower() for defi in defi_terms) else 0
        
        # Infrastructure indicators
        infra_terms = ['proxy', 'forwarder', 'implementation', 'vault', 'wallet', 
                       'weth', 'token', 'account', 'contract']
        features['is_infrastructure'] = 1 if any(infra in entity_name.lower() for infra in infra_terms) else 0
        
        # Length features
        features['name_length'] = len(entity_name)
        features['word_count'] = len(entity_name.split())
        features['has_parentheses'] = 1 if '(' in entity_name else 0
        features['has_dot'] = 1 if '.' in entity_name else 0
        
        return features
    
    def prepare_training_data(self, manual_classification):
        """Prepare training data from manual classification"""
        entities = []
        labels = []
        
        for entity_name, classification in manual_classification.items():
            entities.append(entity_name)
            labels.append(1 if classification['is_defi'] else 0)
        
        return entities, labels
    
    def create_feature_matrix(self, entities):
        """Create feature matrix combining TF-IDF and hand-crafted features"""
        # TF-IDF features
        tfidf_sparse = self.vectorizer.transform(entities)
        tfidf_features = np.asarray(tfidf_sparse.todense())  # <-- Use .todense() then np.asarray()
    
        # Hand-crafted features
        manual_features = []
        for entity in entities:
            features = self.extract_features(entity)
            manual_features.append(list(features.values()))
        
        manual_features = np.array(manual_features)
        
        # Combine features
        combined_features = np.hstack([tfidf_features, manual_features])
        
        return combined_features
    
    def train(self, manual_classification):
        """Train the classifier"""
        entities, labels = self.prepare_training_data(manual_classification)
        
        # Fit vectorizer on training data
        self.vectorizer.fit(entities)
        
        # Create feature matrix
        X = self.create_feature_matrix(entities)
        y = np.array(labels)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        print("Model Performance on Validation Set:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Non-DeFi', 'DeFi']))
        
        # Train on full dataset for final model
        self.classifier.fit(X, y)
        
        return self
    
    def predict(self, entity_name):
        """Predict if an entity is DeFi"""
        # Apply hard rules first
        if entity_name.startswith('@'):
            return False, 1.0  # High confidence it's not DeFi
        
        # Known CEXs
        cex_terms = ['binance', 'coinbase', 'kraken', 'bybit', 'okx', 'kucoin']
        if any(cex in entity_name.lower() for cex in cex_terms):
            return False, 0.95
        
        # Known DeFi protocols
        defi_terms = ['uniswap', 'curve', 'sushi', 'pancake', 'aave', 'compound']
        if any(defi in entity_name.lower() for defi in defi_terms):
            return True, 0.95
        
        # Use model for prediction
        X = self.create_feature_matrix([entity_name])
        prediction = self.classifier.predict(X)[0]
        probability = self.classifier.predict_proba(X)[0]
        
        is_defi = bool(prediction)
        confidence = probability[1] if is_defi else probability[0]
        
        return is_defi, confidence
    
    def classify_all(self, counterparties_df):
        """Classify all counterparties"""
        results = {}
    
        for entity_name in counterparties_df['entity_name'].unique():  # <-- Changed to 'entity_name'
            if pd.isna(entity_name) or entity_name == '':
                continue
        
            is_defi, confidence = self.predict(entity_name)
            results[entity_name] = {
                'is_defi': is_defi,
                'confidence': float(confidence)
            }
    
        return results


def main():#

    # Set up automatic paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manual_classification_path = os.path.join(script_dir, 'manual_classification.json')
    counterparties_path = os.path.join(script_dir, 'counterparties.csv')
    output_file = os.path.join(script_dir, 'classification.json')


    # Load manual classification
    print("Loading manual classification...")
    with open(manual_classification_path, 'r') as f:
        manual_classification = json.load(f)
    
    print(f"Loaded {len(manual_classification)} manually classified entities")
    
    # Load counterparties
    print("\nLoading counterparties...")
    counterparties_df = pd.read_csv(counterparties_path)
    print(f"Loaded {len(counterparties_df)} counterparty records")
    print(f"Unique entities: {counterparties_df['entity_name'].nunique()}")
    
    # Initialize and train classifier
    print("\nTraining classifier...")
    classifier = DeFiClassifier()
    classifier.train(manual_classification)
    
    # Classify all counterparties
    print("\nClassifying all counterparties...")
    classification_results = classifier.classify_all(counterparties_df)
    
    # Calculate statistics
    defi_count = sum(1 for v in classification_results.values() if v['is_defi'])
    non_defi_count = len(classification_results) - defi_count
    
    print(f"\nClassification Results:")
    print(f"Total entities classified: {len(classification_results)}")
    print(f"DeFi entities: {defi_count} ({defi_count/len(classification_results)*100:.1f}%)")
    print(f"Non-DeFi entities: {non_defi_count} ({non_defi_count/len(classification_results)*100:.1f}%)")
    
    # Show some examples
    print("\nSample Classifications:")
    print("-" * 80)
    for i, (entity, classification) in enumerate(list(classification_results.items())[:20]):
        defi_label = "DeFi" if classification['is_defi'] else "Non-DeFi"
        confidence = classification['confidence']
        print(f"{entity:40s} -> {defi_label:10s} (confidence: {confidence:.2f})")
    
    # Validate against manual classification
    print("\nValidation against manual classification:")
    print("-" * 80)
    correct = 0
    total = 0
    mismatches = []
    
    for entity, manual_class in manual_classification.items():
        if entity in classification_results:
            predicted = classification_results[entity]['is_defi']
            actual = manual_class['is_defi']
            total += 1
            
            if predicted == actual:
                correct += 1
            else:
                mismatches.append({
                    'entity': entity,
                    'predicted': predicted,
                    'actual': actual,
                    'confidence': classification_results[entity]['confidence']
                })
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy on manual labels: {accuracy:.2%} ({correct}/{total})")
    
    if mismatches:  
        print(f"\nMisclassified entities ({len(mismatches)}):")
        for mismatch in mismatches[:10]:  # Show first 10
            print(f"  {mismatch['entity']:40s} -> Predicted: {mismatch['predicted']}, "
                  f"Actual: {mismatch['actual']}, Confidence: {mismatch['confidence']:.2f}")
    
    # Save results
    output_file = os.path.join(script_dir, 'classification.json')
    print(f"\nSaving classification results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(classification_results, f, indent=2)
    
    print("Done!")
    
    return classification_results, classifier


if __name__ == "__main__":
    results, classifier = main()