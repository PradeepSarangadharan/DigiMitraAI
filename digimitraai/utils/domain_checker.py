class DomainChecker:
    def __init__(self):
        # Convert all keywords to lowercase for case-insensitive matching
        self.domain_keywords = {
            'aadhaar', 'aadhar', 'adhar', 'uid', 'uidai', 'biometric', 'enrollment', 'enrolment',
            'demographic', 'authentication', 'ekyc', 'kyc', 'resident', 'virtual id', 'identity',
            'card', 'number', 'unique identification', 'address', 'mobile', 'email', 'fingerprint',
            'iris', 'face', 'photo', 'otp', 'masked', 'mandatory', 'optional', 'register',
            'enrollment center', 'aadhaar card', 'demographic', 'biometric', 'bank linking',
            'pan linking', 'mobile number', 'email', 'update', 'correction', 'आधार', 'अदहार'  # Add more transliterated versions
        }
        # Add variations of keywords
        additional_keywords = set()
        for keyword in self.domain_keywords:
            # Add keyword parts if it's a multi-word term
            additional_keywords.update(keyword.split())
        self.domain_keywords.update(additional_keywords)

    def is_domain_relevant(self, query: str) -> tuple[bool, float]:
        """
        Check if query is related to Aadhaar domain
        Returns: (is_relevant, relevance_score)
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Direct checks for common variations
        if any(term in query_lower for term in ['aadhaar', 'aadhar', 'adhar', 'uid', 'uidai']):
            return True, 1.0

        # Word-based matching
        matching_keywords = query_words.intersection(self.domain_keywords)
        
        # If any matches found
        if matching_keywords:
            relevance_score = min(1.0, len(matching_keywords) / len(query_words) + 0.3)  # Added base score
            return True, relevance_score
            
        # Check for partial matches
        for word in query_words:
            if any(keyword in word or word in keyword for keyword in self.domain_keywords):
                return True, 0.7

        return False, 0.0