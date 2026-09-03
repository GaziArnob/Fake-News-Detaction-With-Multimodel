# Optional notebook check: confirms which live services are configured.
from news_guard.service import NewsVerificationService

verification_service = NewsVerificationService()
verification_service.service_status()

