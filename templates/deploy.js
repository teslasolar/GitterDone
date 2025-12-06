/**
 * Deployment Templates
 * Ready-to-use configurations for different hosting platforms
 */

export const DeploymentTemplates = {
  // ============================================
  // GITHUB PAGES
  // ============================================
  'github-pages': {
    name: 'GitHub Pages',
    description: 'Free static hosting with Git LFS for model weights',
    platform: 'github',
    features: ['Free', 'Git LFS support', 'Custom domain', 'HTTPS'],
    limits: {
      maxRepoSize: '5GB',
      maxFileSize: '100MB',
      lfsStorage: '1GB free',
      bandwidth: '100GB/month'
    },

    files: {
      '.github/workflows/deploy.yml': `name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: '.'

      - name: Deploy to GitHub Pages
        uses: actions/deploy-pages@v4
`,

      '.gitattributes': `# Git LFS for model weights
*.bin filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
`,

      'CNAME': '# Add your custom domain here (optional)\n# example: models.yourdomain.com\n'
    },

    setup: [
      'git lfs install',
      'git lfs track "*.bin"',
      'git add .gitattributes',
      'git add -A',
      'git commit -m "Initial model deployment"',
      'git push origin main',
      '# Go to Settings → Pages → Deploy from branch → main'
    ]
  },

  // ============================================
  // VERCEL
  // ============================================
  'vercel': {
    name: 'Vercel',
    description: 'Edge deployment with serverless functions',
    platform: 'vercel',
    features: ['Edge network', 'Serverless API', 'Preview deploys', 'Analytics'],
    limits: {
      maxDeploySize: '100MB',
      serverlessTimeout: '10s',
      bandwidth: '100GB/month'
    },

    files: {
      'vercel.json': `{
  "buildCommand": "echo 'Static site'",
  "outputDirectory": ".",
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "Access-Control-Allow-Origin", "value": "*" },
        { "key": "Cross-Origin-Embedder-Policy", "value": "require-corp" },
        { "key": "Cross-Origin-Opener-Policy", "value": "same-origin" }
      ]
    },
    {
      "source": "/models/(.*).bin",
      "headers": [
        { "key": "Cache-Control", "value": "public, max-age=31536000, immutable" }
      ]
    }
  ],
  "rewrites": [
    { "source": "/api/models", "destination": "/api/models.js" }
  ]
}`,

      'api/models.js': `export default function handler(req, res) {
  res.json({
    models: [
      { id: 'default', name: 'Default Model', url: '/models/default' }
    ]
  })
}
`,

      '.vercelignore': `node_modules
.git
*.log
`
    },

    setup: [
      'npm i -g vercel',
      'vercel login',
      'vercel --prod',
      '# Models > 100MB need external storage (R2, S3, etc.)'
    ]
  },

  // ============================================
  // CLOUDFLARE PAGES
  // ============================================
  'cloudflare-pages': {
    name: 'Cloudflare Pages',
    description: 'Global edge network with R2 storage for large models',
    platform: 'cloudflare',
    features: ['Edge network', 'R2 storage', 'Workers', 'Free tier generous'],
    limits: {
      maxFileSize: '25MB (Pages), Unlimited (R2)',
      requests: '100k/day free',
      bandwidth: 'Unlimited'
    },

    files: {
      '_headers': `/*
  Access-Control-Allow-Origin: *
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Opener-Policy: same-origin

/models/*.bin
  Cache-Control: public, max-age=31536000, immutable
`,

      '_redirects': `# Redirect to R2 for large model files
/models/large/* https://models.r2.yourdomain.com/:splat 200
`,

      'wrangler.toml': `name = "webgpu-model"
compatibility_date = "2024-01-01"

[[r2_buckets]]
binding = "MODELS"
bucket_name = "model-weights"

[site]
bucket = "./"
`
    },

    setup: [
      'npm i -g wrangler',
      'wrangler login',
      'wrangler r2 bucket create model-weights',
      'wrangler r2 object put model-weights/shard_000.bin --file=models/shard_000.bin',
      'wrangler pages deploy .'
    ]
  },

  // ============================================
  // DOCKER
  // ============================================
  'docker': {
    name: 'Docker Container',
    description: 'Self-hosted container with nginx',
    platform: 'docker',
    features: ['Self-hosted', 'Full control', 'Any cloud', 'Offline capable'],

    files: {
      'Dockerfile': `FROM nginx:alpine

# Copy static files
COPY . /usr/share/nginx/html/

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s \\
  CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1
`,

      'nginx.conf': `server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml;

    # CORS headers for WebGPU
    add_header Access-Control-Allow-Origin *;
    add_header Cross-Origin-Embedder-Policy require-corp;
    add_header Cross-Origin-Opener-Policy same-origin;

    # Cache model weights aggressively
    location ~* \\.bin$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }
}
`,

      'docker-compose.yml': `version: '3.8'

services:
  webgpu-model:
    build: .
    ports:
      - "8080:80"
    volumes:
      - ./models:/usr/share/nginx/html/models:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 256M
`,

      '.dockerignore': `node_modules
.git
*.log
.env
`
    },

    setup: [
      'docker build -t webgpu-model .',
      'docker run -p 8080:80 webgpu-model',
      '# Or with docker-compose:',
      'docker-compose up -d'
    ]
  },

  // ============================================
  // AWS S3 + CLOUDFRONT
  // ============================================
  'aws-s3': {
    name: 'AWS S3 + CloudFront',
    description: 'Enterprise-grade hosting with global CDN',
    platform: 'aws',
    features: ['Global CDN', 'Pay-per-use', 'Scalable', 'Enterprise'],

    files: {
      'aws-deploy.sh': `#!/bin/bash
BUCKET_NAME="your-model-bucket"
DISTRIBUTION_ID="your-cloudfront-id"

# Sync files to S3
aws s3 sync . s3://$BUCKET_NAME --exclude ".git/*" --exclude "node_modules/*"

# Set cache headers for model weights
aws s3 cp s3://$BUCKET_NAME/models/ s3://$BUCKET_NAME/models/ \\
  --recursive \\
  --metadata-directive REPLACE \\
  --cache-control "public, max-age=31536000, immutable"

# Invalidate CloudFront cache
aws cloudfront create-invalidation \\
  --distribution-id $DISTRIBUTION_ID \\
  --paths "/*"

echo "Deployed to https://your-domain.cloudfront.net"
`,

      'cloudformation.yml': `AWSTemplateFormatVersion: '2010-09-09'
Description: WebGPU Model Hosting

Resources:
  ModelBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '\${AWS::StackName}-models'
      CorsConfiguration:
        CorsRules:
          - AllowedHeaders: ['*']
            AllowedMethods: [GET, HEAD]
            AllowedOrigins: ['*']

  CloudFrontDistribution:
    Type: AWS::CloudFront::Distribution
    Properties:
      DistributionConfig:
        Origins:
          - DomainName: !GetAtt ModelBucket.RegionalDomainName
            Id: S3Origin
            S3OriginConfig:
              OriginAccessIdentity: ''
        DefaultCacheBehavior:
          TargetOriginId: S3Origin
          ViewerProtocolPolicy: redirect-to-https
          CachePolicyId: 658327ea-f89d-4fab-a63d-7e88639e58f6

Outputs:
  BucketName:
    Value: !Ref ModelBucket
  DistributionDomain:
    Value: !GetAtt CloudFrontDistribution.DomainName
`
    },

    setup: [
      'aws cloudformation create-stack --stack-name webgpu-model --template-body file://cloudformation.yml',
      'chmod +x aws-deploy.sh',
      './aws-deploy.sh'
    ]
  },

  // ============================================
  // HUGGING FACE SPACES
  // ============================================
  'huggingface-spaces': {
    name: 'Hugging Face Spaces',
    description: 'ML-focused hosting with model hub integration',
    platform: 'huggingface',
    features: ['ML community', 'Model versioning', 'Gradio/Streamlit', 'Free tier'],

    files: {
      'README.md': `---
title: WebGPU Model
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: static
pinned: false
---

# WebGPU Model Runner

Run ML models directly in your browser with WebGPU acceleration.
`,

      '.gitattributes': `*.bin filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
`
    },

    setup: [
      'huggingface-cli login',
      'huggingface-cli repo create your-model --type space --space_sdk static',
      'git remote add hf https://huggingface.co/spaces/your-username/your-model',
      'git push hf main'
    ]
  }
}

/**
 * Generate deployment files for a platform
 */
export function generateDeployment(platform, options = {}) {
  const template = DeploymentTemplates[platform]
  if (!template) {
    throw new Error(`Unknown platform: ${platform}`)
  }

  const files = { ...template.files }

  // Customize with options
  if (options.domain) {
    if (files['CNAME']) {
      files['CNAME'] = options.domain + '\n'
    }
  }

  if (options.modelUrl) {
    // Update model URLs in configs
    for (const [name, content] of Object.entries(files)) {
      if (typeof content === 'string') {
        files[name] = content.replace(/\/models\/default/g, options.modelUrl)
      }
    }
  }

  return {
    platform,
    template,
    files,
    setup: template.setup
  }
}

/**
 * Get deployment recommendation based on requirements
 */
export function recommendDeployment(requirements = {}) {
  const { modelSize, budget, selfHosted, needsApi } = requirements

  if (selfHosted) {
    return 'docker'
  }

  if (needsApi) {
    return 'vercel'
  }

  const sizeBytes = parseSize(modelSize || '0')

  if (sizeBytes > 100 * 1024 * 1024) {
    // > 100MB needs LFS or external storage
    if (budget === 'free') {
      return 'github-pages' // with LFS
    }
    return 'cloudflare-pages' // with R2
  }

  if (budget === 'free') {
    return 'github-pages'
  }

  if (budget === 'enterprise') {
    return 'aws-s3'
  }

  return 'vercel'
}

function parseSize(sizeStr) {
  const match = String(sizeStr).match(/^([\d.]+)\s*(KB|MB|GB)?$/i)
  if (!match) return 0
  const num = parseFloat(match[1])
  const unit = (match[2] || 'B').toUpperCase()
  const multipliers = { B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3 }
  return num * multipliers[unit]
}

export default DeploymentTemplates
