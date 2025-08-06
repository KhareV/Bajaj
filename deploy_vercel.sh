#!/bin/bash
# Vercel Deployment Script
# This script deploys the championship AI system to Vercel

echo "🏆 DEPLOYING CHAMPIONSHIP AI SYSTEM TO VERCEL"
echo "=============================================="

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo "🚀 Starting Vercel deployment..."
echo "📝 Make sure to set environment variables in Vercel dashboard"
echo "📄 Copy variables from your local .env file"
echo ""

# Deploy to production
echo "⚡ Deploying to production..."
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo "🏆 Your championship AI system is now live!"
echo ""
echo "📋 NEXT STEPS:"
echo "1. Copy environment variables to Vercel dashboard"
echo "2. Test deployment with test_vercel.py"
echo "3. Submit competition endpoint: https://your-app.vercel.app/hackrx/run"
echo ""
echo "🎯 Competition ready! 🚀"
